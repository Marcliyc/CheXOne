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
from typing import Dict, Iterable, List, Optional, Tuple

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


def _to_sample(image_path: str, solution: str, dataset_name: str, task_name: str, with_reasoning: bool) -> Dict:
    return {
        'images': [image_path],
        'messages': [{'role': 'user', 'content': _build_prompt(task_name, with_reasoning)}],
        'solution': solution,
        'task_name': task_name,
        'dataset_name': dataset_name,
    }


def _find_target_text(row: pd.Series, task: str) -> Optional[str]:
    findings_candidates = ['findings', 'finding', 'report_findings', 'cxr_findings', 'label_findings']
    impression_candidates = ['impression', 'report_impression', 'cxr_impression', 'label_impression']
    generic_candidates = ['report', 'report_text', 'text', 'answer', 'target', 'solution']
    if task == 'findings':
        text = _first_existing(row, findings_candidates + generic_candidates)
    else:
        text = _first_existing(row, impression_candidates + generic_candidates)
    return _clean_text(text)


def _resolve_rex_image_path(row: pd.Series, data_root: Path, mimic_jpg_root: Optional[Path]) -> Optional[str]:
    direct = _first_existing(
        row,
        ['image_path', 'image', 'path', 'img_path', 'jpg_path', 'dicom_path', 'filepath', 'file_path'])
    if direct:
        p = Path(direct)
        if p.is_absolute() and p.exists():
            return str(p)
        candidate = (data_root / p).resolve()
        if candidate.exists():
            return str(candidate)
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


def prepare_rex(args: argparse.Namespace) -> List[Dict]:
    df = _read_table(Path(args.rex_table))
    samples: List[Dict] = []
    task_name = 'Findings Generation' if args.task == 'findings' else 'Impression Generation'
    for _, row in df.iterrows():
        text = _find_target_text(row, args.task)
        if text is None:
            continue
        image_path = _resolve_rex_image_path(
            row=row, data_root=Path(args.data_root), mimic_jpg_root=Path(args.mimic_jpg_root)
            if args.mimic_jpg_root else None)
        if image_path is None:
            continue
        samples.append(_to_sample(image_path, text, 'ReXGradient-160K', task_name, args.with_reasoning))
    return samples


def _build_mimic_image_path(mimic_jpg_root: Path, subject_id, study_id, dicom_id) -> Path:
    subject_num = str(subject_id).replace('p', '')
    study_num = str(study_id).replace('s', '')
    return mimic_jpg_root / 'files' / f'p{subject_num[:2]}' / f'p{subject_num}' / f's{study_num}' / f'{dicom_id}.jpg'


def _read_mimic_report(reports_root: Path, subject_id, study_id, task: str) -> Optional[str]:
    subject_num = str(subject_id).replace('p', '')
    study_num = str(study_id).replace('s', '')
    report_path = reports_root / f'p{subject_num[:2]}' / f'p{subject_num}' / f's{study_num}.txt'
    if not report_path.exists():
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
    for _, row in metadata.iterrows():
        dicom_id = _first_existing(row, ['dicom_id'])
        subject_id = _first_existing(row, ['subject_id'])
        study_id = _first_existing(row, ['study_id'])
        if not dicom_id or not subject_id or not study_id:
            continue
        image_path = _build_mimic_image_path(mimic_jpg_root, subject_id, study_id, dicom_id)
        if not image_path.exists():
            continue
        text = _read_mimic_report(reports_root, subject_id, study_id, args.task)
        if text is None:
            continue
        samples.append(_to_sample(str(image_path.resolve()), text, 'MIMIC-CXR', task_name, args.with_reasoning))
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
    parser.add_argument('--mimic-jpg-root', type=str, default='data/mimic-cxr-jpg')
    parser.add_argument('--mimic-metadata', type=str, default='data/mimic-cxr-jpg/mimic-cxr-2.0.0-metadata.csv.gz')
    parser.add_argument('--mimic-split', type=str, default='data/mimic-cxr-jpg/mimic-cxr-2.0.0-split.csv.gz')
    parser.add_argument('--split', choices=['train', 'validate', 'test'], default='train')
    parser.add_argument('--mimic-reports-root', type=str, default='data/mimic-cxr')

    parser.add_argument('--prepare-rex', action='store_true')
    parser.add_argument('--rex-table', type=str, default='data/ReXGradient-160K/train.parquet')

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
        rex_rows = prepare_rex(args)
        random.shuffle(rex_rows)
        if args.max_samples_per_dataset > 0:
            rex_rows = rex_rows[:args.max_samples_per_dataset]
        print(f'[prepare-rex] collected: {len(rex_rows)}')
        all_rows.extend(rex_rows)

    random.shuffle(all_rows)
    out_name = f'{args.output_prefix}_{args.task}_grpo.jsonl'
    out_path = Path(args.data_root) / 'prepared' / out_name
    write_jsonl(out_path, all_rows)
    print(f'[done] wrote {len(all_rows)} samples to {out_path}')


if __name__ == '__main__':
    main()
