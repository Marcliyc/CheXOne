# CheXOne dataset preparation (MIMIC-CXR + ReXGradient-160K)

This folder contains a helper script to convert downloaded datasets into the JSONL format used by SWIFT GRPO/SFT pipelines.

## Output format

Each output row contains:

```json
{
  "images": ["/path/to/image.jpg"],
  "messages": [{"role": "user", "content": "<image>Write the findings for this chest X-ray."}],
  "solution": "Ground-truth report text",
  "task_name": "Findings Generation",
  "dataset_name": "MIMIC-CXR"
}
```

- `messages` + `images` are model inputs.
- `solution` is passed to reward functions in GRPO.

## Script

`prepare_cxr_datasets.py`

### 1) Prepare MIMIC-CXR only

```bash
python examples/train/chexone/data_prep/prepare_cxr_datasets.py \
  --prepare-mimic \
  --task findings \
  --mimic-jpg-root data/MIMIC-CXR \
  --mimic-metadata data/MIMIC-CXR/mimic-cxr-2.1.0-metadata.csv.gz \
  --mimic-split data/MIMIC-CXR/mimic-cxr-2.1.0-split.csv.gz \
  --mimic-reports-root data/MIMIC-CXR \
  --data-root data
```

> Note: `mimic-cxr-jpg` provides images/metadata; report text is read from `--mimic-reports-root` (from MIMIC-CXR report files).  
> The script handles both `<root>/files/...` and `<root>/2.1.0/files/...` layouts.
> It also handles your shown structure such as `images/mimic-cxr-jpg-2.1.0/mimic-cxr-jpg-2.1.0.physionet.org/files/...`.

### 2) Prepare ReXGradient-160K only

```bash
python examples/train/chexone/data_prep/prepare_cxr_datasets.py \
  --prepare-rex \
  --task findings \
  --rex-splits train \
  --rex-metadata-dir data/ReXGradient/metadata \
  --rex-image-root data/ReXGradient/deid_png \
  --mimic-jpg-root data/MIMIC-CXR \
  --data-root data
```

The script can read `parquet`, `csv`, `csv.gz`, `json`, or `jsonl` for `--rex-table`.
If `deid_png` has not been extracted yet (only `deid_png.part*` exists), extract it first.

To prepare validation/test as well:

```bash
python examples/train/chexone/data_prep/prepare_cxr_datasets.py \
  --prepare-rex \
  --task findings \
  --rex-splits train,valid,test \
  --rex-metadata-dir data/ReXGradient/metadata \
  --rex-image-root data/ReXGradient/deid_png \
  --write-rex-split-files \
  --output-prefix chexone_rex \
  --data-root data
```

This writes:
- `data/prepared/chexone_rex_findings_grpo_rex_train.jsonl`
- `data/prepared/chexone_rex_findings_grpo_rex_valid.jsonl`
- `data/prepared/chexone_rex_findings_grpo_rex_test.jsonl`

### 3) Prepare and merge both

```bash
python examples/train/chexone/data_prep/prepare_cxr_datasets.py \
  --prepare-mimic \
  --prepare-rex \
  --task findings \
  --rex-splits train,valid,test \
  --rex-metadata-dir data/ReXGradient/metadata \
  --rex-image-root data/ReXGradient/deid_png \
  --mimic-jpg-root data/MIMIC-CXR \
  --mimic-metadata data/MIMIC-CXR/mimic-cxr-2.1.0-metadata.csv.gz \
  --mimic-split data/MIMIC-CXR/mimic-cxr-2.1.0-split.csv.gz \
  --mimic-reports-root data/MIMIC-CXR \
  --output-prefix chexone_mix \
  --max-samples-per-dataset 50000 \
  --data-root data
```

Output path:

`data/prepared/<output-prefix>_<task>_grpo.jsonl`

## 4) Run GRIT-format RLHF on Slurm + Apptainer

Use:

`examples/train/chexone/train_script/6_rex160k_grit_radcliq_slurm_apptainer.sh`

This script avoids common shell pitfalls (unclosed `if`/`fi`, trailing `\`) and sets robust plugin path resolution + container bind paths for host absolute paths.
It also prefers `SLURM_SUBMIT_DIR` when resolving plugin paths under `sbatch` spool execution.
For `StanfordAIMI/CheXOne`, the GRIT training scripts default `USE_HF=1`/`--use_hf` so Swift downloads from Hugging Face instead of ModelScope.
