#!/usr/bin/env python3
"""Prepare MIMIC-CXR and ReXGradient-160K into SWIFT-compatible JSONL.

Output schema (per line):
{
  "images": ["/abs/or/rel/path/to/image.jpg"],
  "messages": [{"role": "user", "content": "<image>...prompt..."}],
  "solution": "...ground truth text...",
  "task_name": "Findings Generation" | "Impression Generation",
  "dataset_name": "MIMIC-CXR" | "ReXGradient-160K"
}

`solution` and `task_name` are kept for GRPO reward functions.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


def _read_table(path: Path) -> pd.DataFrame:
    suffixes = ''.join(path.suffixes).lower()
    if suffixes.endswith('.jsonl'):
        return pd.read_json(path, lines=True)
    if suffixes.endswith('.json'):
        return pd.read_json(path)
    if suffixes.endswith('.csv') or suffixes.endswith('.csv.gz'):
        return pd.read_csv(path)
    if suffixes.endswith('.parquet'):
        return pd.read_parquet(path)
    raise ValueError(f'Unsupported table format: {path}')


def _clean_text(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {'nan', 'none', 'null'}:
        return None
    return text


def _first_existing(row: pd.Series, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in row and _clean_text(row[c]) is not None:
            return _clean_text(row[c])
    return None


def _build_prompt(task_name: str, with_reasoning: bool) -> str:
    if task_name == 'Impression Generation':
        if with_reasoning:
            return '<image>Write the impression for this chest X-ray. Reason step by step and put final answer in \\boxed{}.'
        return '<image>Write the impression for this chest X-ray.'
    if with_reasoning:
        return '<image>Write the findings for this chest X-ray. Reason step by step and put final answer in \\boxed{}.'
    return '<image>Write the findings for this chest X-ray.'


def _candidate_files_roots(root: Path, version_hint: str = '2.1.0') -> List[Path]:
    return [
        root,
        root / 'files',
        root / version_hint,
        root / version_hint / 'files',
        root / f'mimic-cxr-jpg-{version_hint}',
        root / f'mimic-cxr-jpg-{version_hint}' / f'mimic-cxr-jpg-{version_hint}.physionet.org',
        root / f'mimic-cxr-jpg-{version_hint}' / f'mimic-cxr-jpg-{version_hint}.physionet.org' / 'files',
        root / 'images' / f'mimic-cxr-jpg-{version_hint}',
        root / 'images' / f'mimic-cxr-jpg-{version_hint}' / f'mimic-cxr-jpg-{version_hint}.physionet.org',
        root / 'images' / f'mimic-cxr-jpg-{version_hint}' / f'mimic-cxr-jpg-{version_hint}.physionet.org' / 'files',
        root / f'mimic-cxr-{version_hint}.physionet.org',
        root / f'mimic-cxr-{version_hint}.physionet.org' / 'files',
        root / 'data' / f'mimic-cxr-{version_hint}.physionet.org',
        root / 'data' / f'mimic-cxr-{version_hint}.physionet.org' / 'files',
    ]


def _resolve_files_root(root: Path, version_hint: str = '2.1.0') -> Path:
    """Resolve MIMIC files root across common unpacked layouts."""
    candidates = _candidate_files_roots(root, version_hint)
    for c in candidates:
        if c.exists() and c.is_dir():
            if any((c / f'p{i:02d}').exists() for i in range(10, 20)):
                return c
    return root


def _to_sample(image_path: str,
               solution: str,
               dataset_name: str,
               task_name: str,
               with_reasoning: bool,
               dataset_split: Optional[str] = None) -> Dict:
    sample = {
        'images': [image_path],
        'messages': [{'role': 'user', 'content': _build_prompt(task_name, with_reasoning)}],
        'solution': solution,
        'task_name': task_name,
        'dataset_name': dataset_name,
    }
    if dataset_split is not None:
        sample['dataset_split'] = dataset_split
    return sample


def _find_target_text(row: pd.Series, task: str) -> Optional[str]:
    findings_candidates = [
        'findings', 'finding', 'Findings', 'FINDINGS', 'report_findings', 'findings_text', 'findings_section',
        'cxr_findings', 'label_findings'
    ]
    impression_candidates = [
        'impression', 'Impression', 'IMPRESSION', 'report_impression', 'impression_text', 'impression_section',
        'cxr_impression', 'label_impression'
    ]
    generic_candidates = ['report', 'report_text', 'text', 'answer', 'target', 'solution']
    if task == 'findings':
        text = _first_existing(row, findings_candidates + generic_candidates)
    else:
        text = _first_existing(row, impression_candidates + generic_candidates)
    return _clean_text(text)


def _resolve_rex_image_path(row: pd.Series,
                            data_root: Path,
                            mimic_jpg_root: Optional[Path],
                            rex_image_root: Optional[Path],
                            split: Optional[str] = None) -> Optional[str]:
    def _try_existing(paths: List[Path]) -> Optional[str]:
        for candidate in paths:
            if candidate.exists():
                return str(candidate.resolve())
        return None

    direct = _first_existing(
        row,
        [
            'image_path', 'image', 'path', 'img_path', 'jpg_path', 'png_path', 'dicom_path', 'filepath', 'file_path',
            'image_file', 'image_filename', 'filename', 'file_name', 'image_name', 'img_name', 'img', 'image_id', 'id'
        ])
    if direct:
        p = Path(direct)
        if p.is_absolute() and p.exists():
            return str(p)
        rex_candidates: List[Path] = []
        if rex_image_root is not None:
            split_paths = [Path(split)] if split else []
            split_paths.append(Path(''))
            base_names = [p]
            if p.suffix == '':
                base_names.extend([Path(f'{p}.png'), Path(f'{p}.jpg'), Path(f'{p}.jpeg')])
            for split_path in split_paths:
                for name in base_names:
                    rex_candidates.append((rex_image_root / split_path / name).resolve())
            found = _try_existing(rex_candidates)
            if found:
                return found
        data_candidates = [(data_root / p).resolve()]
        if p.suffix == '':
            data_candidates.extend([
                (data_root / f'{p}.png').resolve(),
                (data_root / f'{p}.jpg').resolve(),
                (data_root / f'{p}.jpeg').resolve(),
            ])
        found = _try_existing(data_candidates)
        if found:
            return found
    dicom_id = _first_existing(row, ['dicom_id'])
    subject_id = _first_existing(row, ['subject_id'])
    study_id = _first_existing(row, ['study_id'])
    if dicom_id and subject_id and study_id and mimic_jpg_root is not None:
        subject_num = str(subject_id)
        if subject_num.startswith('p'):
            subject_num = subject_num[1:]
        p_prefix = f"p{subject_num[:2]}"
        p_subject = f"p{subject_num}"
        s_study = f"s{str(study_id).lstrip('s')}"
        img = mimic_jpg_root / 'files' / p_prefix / p_subject / s_study / f'{dicom_id}.jpg'
        if img.exists():
            return str(img.resolve())
    return None


def _resolve_rex_table_for_split(args: argparse.Namespace, split: str) -> Path:
    if args.rex_table:
        return Path(args.rex_table)
    metadata_dir = Path(args.rex_metadata_dir)
    candidates = [
        metadata_dir / f'{split}_metadata.csv',
        metadata_dir / f'{split}_metadata.json',
        metadata_dir / f'{split}_metadata.jsonl',
        metadata_dir / f'{split}_metadata.parquet',
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f'Cannot find ReXGradient metadata for split={split} in {metadata_dir}')


def prepare_rex(args: argparse.Namespace, split: str = 'train') -> List[Dict]:
    table_path = _resolve_rex_table_for_split(args, split)
    print(f'[prepare-rex] loading split={split} from {table_path}')
    df = _read_table(table_path)
    if args.rex_image_root:
        rex_root = Path(args.rex_image_root)
        if not rex_root.exists():
            part_files = sorted(rex_root.parent.glob(f'{rex_root.name}.part*'))
            if part_files:
                print(
                    f'[prepare-rex][hint] {rex_root} not found, but archive parts exist '
                    f'({part_files[0].name} ... {part_files[-1].name}). '
                    f'Please merge/extract before running.')
    samples: List[Dict] = []
    missing_text = 0
    missing_image = 0
    task_name = 'Findings Generation' if args.task == 'findings' else 'Impression Generation'
    for _, row in df.iterrows():
        text = _find_target_text(row, args.task)
        if text is None:
            missing_text += 1
            continue
        image_path = _resolve_rex_image_path(
            row=row,
            data_root=Path(args.data_root),
            mimic_jpg_root=Path(args.mimic_jpg_root) if args.mimic_jpg_root else None,
            rex_image_root=Path(args.rex_image_root) if args.rex_image_root else None,
            split=split)
        if image_path is None:
            missing_image += 1
            continue
        samples.append(
            _to_sample(
                image_path=image_path,
                solution=text,
                dataset_name='ReXGradient-160K',
                task_name=task_name,
                with_reasoning=args.with_reasoning,
                dataset_split=split))
    if missing_text > 0 or missing_image > 0:
        print(
            f'[prepare-rex] split={split} skipped rows: '
            f'missing_text={missing_text}, missing_image={missing_image}')
    return samples


def _build_mimic_image_path(mimic_jpg_root: Path, subject_id, study_id, dicom_id) -> Path:
    subject_num = str(subject_id).replace('p', '')
    study_num = str(study_id).replace('s', '')
    for files_root in _candidate_files_roots(mimic_jpg_root):
        p = files_root / f'p{subject_num[:2]}' / f'p{subject_num}' / f's{study_num}' / f'{dicom_id}.jpg'
        if p.exists():
            return p
    files_root = _resolve_files_root(mimic_jpg_root)
    return files_root / f'p{subject_num[:2]}' / f'p{subject_num}' / f's{study_num}' / f'{dicom_id}.jpg'


def _read_mimic_report(reports_root: Path, subject_id, study_id, task: str) -> Optional[str]:
    subject_num = str(subject_id).replace('p', '')
    study_num = str(study_id).replace('s', '')
    report_path = None
    for files_root in _candidate_files_roots(reports_root):
        p = files_root / f'p{subject_num[:2]}' / f'p{subject_num}' / f's{study_num}.txt'
        if p.exists():
            report_path = p
            break
    if report_path is None:
        return None
    txt = report_path.read_text(encoding='utf-8', errors='ignore')
    if task == 'findings':
        start_keys = ['FINDINGS:', 'FINDING:']
        stop_keys = ['IMPRESSION:', 'CONCLUSION:']
    else:
        start_keys = ['IMPRESSION:', 'CONCLUSION:']
        stop_keys = ['RECOMMENDATION:', 'RECOMMENDATIONS:']

    text_upper = txt.upper()
    start_idx = None
    for k in start_keys:
        idx = text_upper.find(k)
        if idx >= 0:
            start_idx = idx + len(k)
            break
    if start_idx is None:
        return None
    end_idx = len(txt)
    for k in stop_keys:
        idx = text_upper.find(k, start_idx)
        if idx >= 0:
            end_idx = min(end_idx, idx)
    return _clean_text(txt[start_idx:end_idx])


def prepare_mimic(args: argparse.Namespace) -> List[Dict]:
    metadata = _read_table(Path(args.mimic_metadata))
    if args.mimic_split:
        split_df = _read_table(Path(args.mimic_split))
        split_col = 'split' if 'split' in split_df.columns else split_df.columns[-1]
        split_df = split_df[split_df[split_col] == args.split]
        keys = [c for c in ['subject_id', 'study_id', 'dicom_id'] if c in split_df.columns]
        metadata = metadata.merge(split_df[keys], on=keys, how='inner')

    task_name = 'Findings Generation' if args.task == 'findings' else 'Impression Generation'
    samples: List[Dict] = []
    mimic_jpg_root = Path(args.mimic_jpg_root)
    reports_root = Path(args.mimic_reports_root)
    missing_img = 0
    missing_txt = 0
    for _, row in metadata.iterrows():
        dicom_id = _first_existing(row, ['dicom_id'])
        subject_id = _first_existing(row, ['subject_id'])
        study_id = _first_existing(row, ['study_id'])
        if not dicom_id or not subject_id or not study_id:
            continue
        image_path = _build_mimic_image_path(mimic_jpg_root, subject_id, study_id, dicom_id)
        if not image_path.exists():
            missing_img += 1
            continue
        text = _read_mimic_report(reports_root, subject_id, study_id, args.task)
        if text is None:
            missing_txt += 1
            continue
        samples.append(_to_sample(str(image_path.resolve()), text, 'MIMIC-CXR', task_name, args.with_reasoning))
    print(f'[prepare-mimic] skipped_missing_image={missing_img}, skipped_missing_report={missing_txt}')
    return samples


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare MIMIC-CXR/ReXGradient-160K for CheXOne training.')
    parser.add_argument('--data-root', type=str, default='data', help='Base data directory.')
    parser.add_argument('--task', choices=['findings', 'impression'], default='findings')
    parser.add_argument('--with-reasoning', action='store_true', help='Use reasoning-style user prompt.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-samples-per-dataset', type=int, default=0, help='0 means no limit.')
    parser.add_argument('--output-prefix', type=str, default='chexone')

    parser.add_argument('--prepare-mimic', action='store_true')
    parser.add_argument('--mimic-jpg-root', type=str, default='data/MIMIC-CXR')
    parser.add_argument('--mimic-metadata', type=str, default='data/MIMIC-CXR/mimic-cxr-2.1.0-metadata.csv.gz')
    parser.add_argument('--mimic-split', type=str, default='data/MIMIC-CXR/mimic-cxr-2.1.0-split.csv.gz')
    parser.add_argument('--split', choices=['train', 'validate', 'test'], default='train')
    parser.add_argument('--mimic-reports-root', type=str, default='data/MIMIC-CXR')

    parser.add_argument('--prepare-rex', action='store_true')
    parser.add_argument('--rex-table', type=str, default=None, help='Optional explicit ReX table file path.')
    parser.add_argument('--rex-metadata-dir', type=str, default='data/ReXGradient/metadata')
    parser.add_argument('--rex-splits', type=str, default='train', help='Comma-separated splits, e.g. train,valid,test')
    parser.add_argument('--rex-image-root', type=str, default='data/ReXGradient/deid_png')
    parser.add_argument('--write-rex-split-files', action='store_true', help='Write separate output files per ReX split.')

    args = parser.parse_args()
    random.seed(args.seed)

    if not args.prepare_mimic and not args.prepare_rex:
        raise ValueError('Please set at least one of --prepare-mimic or --prepare-rex.')

    all_rows: List[Dict] = []
    if args.prepare_mimic:
        mimic_rows = prepare_mimic(args)
        random.shuffle(mimic_rows)
        if args.max_samples_per_dataset > 0:
            mimic_rows = mimic_rows[:args.max_samples_per_dataset]
        print(f'[prepare-mimic] collected: {len(mimic_rows)}')
        all_rows.extend(mimic_rows)

    if args.prepare_rex:
        rex_splits = [s.strip() for s in args.rex_splits.split(',') if s.strip()]
        for split in rex_splits:
            rex_rows = prepare_rex(args, split=split)
            random.shuffle(rex_rows)
            if args.max_samples_per_dataset > 0:
                rex_rows = rex_rows[:args.max_samples_per_dataset]
            print(f'[prepare-rex] collected split={split}: {len(rex_rows)}')
            all_rows.extend(rex_rows)
            if args.write_rex_split_files:
                split_out = Path(args.data_root) / 'prepared' / f'{args.output_prefix}_{args.task}_grpo_rex_{split}.jsonl'
                write_jsonl(split_out, rex_rows)
                print(f'[prepare-rex] wrote split file: {split_out}')

    random.shuffle(all_rows)
    out_name = f'{args.output_prefix}_{args.task}_grpo.jsonl'
    out_path = Path(args.data_root) / 'prepared' / out_name
    write_jsonl(out_path, all_rows)
    print(f'[done] wrote {len(all_rows)} samples to {out_path}')


if __name__ == '__main__':
    main()
