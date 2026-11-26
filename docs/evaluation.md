# Evaluation

[← Back to README](../README.md) &nbsp;·&nbsp; [Method and protocol](method.md)

Run the commands below from the repository root after completing [installation](../README.md#installation).

## Benchmark support

| Benchmark | CLI name | Data interface | Default split |
| :--- | :--- | :--- | :--- |
| VQAv2 | `VQAv2` | Native loader | `val` |
| TextVQA | `TextVQA` | Native loader | `val` |
| VizWiz | `VizWiz` | Native loader | `val` |
| MMBench | `MMBench` | Native loader | `dev` |
| ScienceQA | `ScienceQA` | Native loader | `test` |
| POPE | `POPE` | Native loader | `adversarial` |
| MME | `MME` | Native loader | `test` |
| MMMU | `MMMU` | JSON/JSONL manifest | `validation` |
| GQA | `GQA` | JSON/JSONL manifest | `testdev_balanced` |
| MM-Vet | `MMVet` | JSON/JSONL manifest | `test` |

Model weights and datasets are downloaded separately. The model ID can be replaced with a local snapshot:

```bash
export MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
```

## Native-loader evaluation

For example, run MMBench with a 64-token budget:

```bash
python scripts/evaluate.py \
  --model-path "$MODEL_PATH" \
  --dataset MMBench \
  --data-root /path/to/MMBench \
  --split dev \
  --mmbench-lang en \
  --budget 64 \
  --device cuda:0 \
  --output-dir outputs/mmbench_en_b64
```

For POPE, set `--image-dir` when the images are stored separately from the annotation files.

## Normalized manifests

MMMU, GQA, and MM-Vet require a normalized JSON/JSONL manifest. Each sample contains:

| Field | Description |
| :--- | :--- |
| `id` | Sample identifier |
| `image_path` | Path to the sample image |
| `question` | Question text |
| `answer` or `answers` | One reference answer or a list of answers |
| `options` | For multiple-choice questions: a list of `[label, text]` pairs |

For JSONL, write one complete JSON object per line. One record looks like:

```json
{"id": "sample-001", "image_path": "/path/to/image.jpg", "question": "What is shown in the image?", "answer": "a boat"}
```

Run a manifest-backed benchmark with:

```bash
python scripts/evaluate.py \
  --model-path "$MODEL_PATH" \
  --dataset MMMU \
  --manifest /path/to/mmmu_manifest.jsonl \
  --budget 64 \
  --device cuda:0 \
  --output-dir outputs/mmmu_b64
```

## Runtime options

| Option | Default / accepted values | Purpose |
| :--- | :--- | :--- |
| `--budget` | Required: `32`, `64`, `128`, `192` | Number of visual tokens to retain |
| `--grid-size` | `4`, `5`, `8`, `9` by budget | Override the default coarse grid |
| `--split` | See the benchmark table | Dataset split |
| `--mmbench-lang` | `en`; also `cn`, `cc` | MMBench language |
| `--max-images` | `0` (all samples) | Limit the number of evaluated images |
| `--sample-offset` | `0` | Starting sample offset |
| `--max-new-tokens` | `1024` | Generation limit |
| `--seed` | `42` | Random seed |
| `--device` | `cuda:0` | Model device |

List all options with `python scripts/evaluate.py --help`.

## Outputs and scoring

| Output | Contents |
| :--- | :--- |
| `summary.json` | Aggregate metrics and run metadata |
| `per_sample.csv` | Predictions, per-sample scores, selected indices, regional budgets, and timing |
| `run_config.json` | Model and evaluation parameters |
| `image_cache/` | Decoded images for parquet-backed datasets, when required |

MM-Vet uses an external judge and therefore receives no local score. Submit the predictions in `per_sample.csv` to the official evaluator. Official leaderboard scripts remain authoritative for each benchmark.

## Reproducibility checks

The released settings are recorded in [configs/qwen2_5_vl_7b.json](../configs/qwen2_5_vl_7b.json). Run the allocator regression checks without model weights or CUDA:

```bash
python -m unittest discover -s tests -v
```

<details>
<summary>Metadata check for an anonymized archive</summary>

A separate maintenance script checks for common identity and machine-specific strings:

```bash
python scripts/check_anonymity.py
```

</details>
