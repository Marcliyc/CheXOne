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
  --mimic-jpg-root data/mimic-cxr-jpg \
  --mimic-metadata data/mimic-cxr-jpg/mimic-cxr-2.0.0-metadata.csv.gz \
  --mimic-split data/mimic-cxr-jpg/mimic-cxr-2.0.0-split.csv.gz \
  --mimic-reports-root data/mimic-cxr \
  --data-root data
```

> Note: `mimic-cxr-jpg` provides images/metadata; report text is read from `--mimic-reports-root` (from MIMIC-CXR report files).

### 2) Prepare ReXGradient-160K only

```bash
python examples/train/chexone/data_prep/prepare_cxr_datasets.py \
  --prepare-rex \
  --task findings \
  --rex-table data/ReXGradient-160K/train.parquet \
  --mimic-jpg-root data/mimic-cxr-jpg \
  --data-root data
```

The script can read `parquet`, `csv`, `csv.gz`, `json`, or `jsonl` for `--rex-table`.

### 3) Prepare and merge both

```bash
python examples/train/chexone/data_prep/prepare_cxr_datasets.py \
  --prepare-mimic \
  --prepare-rex \
  --task findings \
  --rex-table data/ReXGradient-160K/train.parquet \
  --mimic-jpg-root data/mimic-cxr-jpg \
  --mimic-metadata data/mimic-cxr-jpg/mimic-cxr-2.0.0-metadata.csv.gz \
  --mimic-split data/mimic-cxr-jpg/mimic-cxr-2.0.0-split.csv.gz \
  --mimic-reports-root data/mimic-cxr \
  --output-prefix chexone_mix \
  --max-samples-per-dataset 50000 \
  --data-root data
```

Output path:

`data/prepared/<output-prefix>_<task>_grpo.jsonl`
