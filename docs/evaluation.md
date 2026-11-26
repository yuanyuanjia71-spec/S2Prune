# Evaluation

[← Back to README](../README.md)

Commands below are run from the repository root, after following the [installation instructions](../README.md#installation). The released backbone, image size, and pruning point are described in the [method guide](method.md).


The evaluator supports two data interfaces. Use the CLI dataset name `MMVet` for the MM-Vet benchmark.

| Interface | Benchmarks |
| --- | --- |
| Native dataset loader | VQAv2, TextVQA, VizWiz, MMBench, ScienceQA, POPE, MME |
| Normalized JSON/JSONL manifest | MMMU, GQA, MM-Vet (`--dataset MMVet`) |

Run `python scripts/evaluate.py --help` for every option.

## Native-loader example

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

For POPE, use `--image-dir` when the COCO images are stored separately from the annotation files.

## Manifest example

Each JSON or JSONL row must contain `id`, `image_path`, `question`, and either `answer` or `answers`. Multiple-choice rows additionally use `options` as `[label, text]` pairs.

```json
{
  "id": "sample-001",
  "image_path": "/path/to/image.jpg",
  "question": "What is shown in the image?",
  "answer": "a boat"
}
```

```bash
python scripts/evaluate.py \
  --model-path "$MODEL_PATH" \
  --dataset MMMU \
  --manifest /path/to/mmmu_manifest.jsonl \
  --budget 64 \
  --device cuda:0 \
  --output-dir outputs/mmmu_b64
```

## Outputs

| Output | Contents |
| --- | --- |
| `per_sample.csv` | Predictions, local scores, selected indices, regional budgets, and timing |
| `summary.json` | Aggregate local metric and run metadata |
| `run_config.json` | Complete command-line configuration |
| `image_cache/` | Images decoded from parquet-backed datasets when required |

MM-Vet intentionally receives no local score because its official evaluation uses an external judge. Use `per_sample.csv` with the official evaluator. Official leaderboard scripts remain authoritative for every benchmark.

## Default splits

The evaluator uses these defaults when `--split` is omitted.

| Benchmark / CLI name | Default split |
| :--- | :--- |
| `VQAv2` | `val` |
| `TextVQA` | `val` |
| `VizWiz` | `val` |
| `MMBench` | `dev` |
| `ScienceQA` | `test` |
| `POPE` | `adversarial` |
| `MME` | `test` |
| `MMMU` | `validation` |
| `GQA` | `testdev_balanced` |
| `MMVet` | `test` |

## Reproducibility Checks

The allocator tests do not require model weights or benchmark data:

```bash
python -m unittest discover -s tests -v
```

Before publishing a release or archive, scan the repository for common identity and machine-specific strings:

```bash
python scripts/check_anonymity.py
```

The released model and budget configuration is recorded in [`configs/qwen2_5_vl_7b.json`](../configs/qwen2_5_vl_7b.json).
