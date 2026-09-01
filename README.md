# S²Prune

<p align="center">
  <strong>Training-free, structure-aware visual-token pruning for multimodal large language models</strong>
</p>

<p align="center">
  Reference implementation for Qwen2.5-VL-7B-Instruct
</p>

<p align="center">
  <img src="assets/s2prune_framework.png" alt="S2Prune framework overview" width="100%">
</p>

S²Prune reduces the visual sequence processed by a multimodal LLM without training an auxiliary selector. It separates **where to spend the token budget** from **which local tokens to retain**:

1. **Structural-density allocation** assigns more capacity to visually complex regions.
2. **Response-aware local sampling** keeps the most informative token in each allocated local cell.

The released implementation physically shortens the sequence after decoder Layer 0 while preserving text tokens, special tokens, original token order, and M-RoPE coordinates.

## Highlights

- **Training-free:** no additional model, fine-tuning, or learned scoring head.
- **Exact token budgets:** retain exactly `B ∈ {32, 64, 128, 192}` tokens from the original 576 visual tokens.
- **Content-adaptive allocation:** distribute the residual budget in proportion to regional Laplacian complexity.
- **Early response-aware selection:** use representation change across decoder Layer 0 to choose local representatives.
- **Physical sequence pruning:** consistently gather hidden states, attention masks, position IDs, and the Layer-0 KV cache.
- **Evaluation-ready:** loaders or normalized manifests for ten common multimodal benchmarks.

## Method

The released protocol resizes each image to `672 × 672`. Qwen2.5-VL PatchMerger then produces a `24 × 24` grid containing 576 decoder-visible visual tokens.

### 1. Allocate tokens by structural density

For each coarse region `g`, S²Prune measures structural complexity with the variance of the image Laplacian:

```text
c_g = Var(Laplacian(I_g))
```

Every region receives one token first. The remaining budget is distributed according to the per-image min-max normalized complexity scores using capacity-aware largest-remainder allocation.

| Visual budget `B` | Coarse grid | Regions | Retained visual tokens |
| ---: | :---: | ---: | ---: |
| 32 | 4 × 4 | 16 | 5.6% |
| 64 | 5 × 5 | 25 | 11.1% |
| 128 | 8 × 8 | 64 | 22.2% |
| 192 | 9 × 9 | 81 | 33.3% |

### 2. Select one representative per local cell

Each region is recursively partitioned into exactly `B_g` non-overlapping cells. After the full visual sequence passes through decoder Layer 0, the early representation change (ERC) score for visual token `i` is

```text
s_i = ||h_i^1 - h_i^0||_2
```

The maximum-ERC token is retained in each cell. Selected indices are sorted into their original decoder-visible order, and physical sequence deletion is applied before decoder Layer 1.

## Installation

The reported environment used Python 3.8.10, CUDA 12.1, PyTorch 2.4.1, and Transformers 4.49.0. Exact package versions are pinned because the Qwen2.5-VL decoder and cache APIs differ across Transformers releases.

```bash
git clone https://github.com/yuanyuanjia71-spec/S2Prune.git
cd S2Prune

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Model weights and benchmark data are not included in this repository.

## Quick Start

Use the Hugging Face model identifier or an equivalent local snapshot:

```bash
export MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct
```

Run a one-image smoke test across every released token budget:

```bash
python scripts/smoke_test.py \
  --model-path "$MODEL_PATH" \
  --image /path/to/example.jpg \
  --device cuda:0
```

The token-count checks should include:

```text
PASS B=32  grid=4x4 visual=576->32
PASS B=64  grid=5x5 visual=576->64
PASS B=128 grid=8x8 visual=576->128
PASS B=192 grid=9x9 visual=576->192
```

## Evaluation

The evaluator supports two data interfaces.

| Interface | Benchmarks |
| --- | --- |
| Native dataset loader | VQAv2, TextVQA, VizWiz, MMBench, ScienceQA, POPE, MME |
| Normalized JSON/JSONL manifest | MMMU, GQA, MM-Vet |

Run `python scripts/evaluate.py --help` for every option.

### Native-loader example

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

### Manifest example

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

### Outputs

| Output | Contents |
| --- | --- |
| `per_sample.csv` | Predictions, local scores, selected indices, regional budgets, and timing |
| `summary.json` | Aggregate local metric and run metadata |
| `run_config.json` | Complete command-line configuration |
| `image_cache/` | Images decoded from parquet-backed datasets when required |

MM-Vet intentionally receives no local score because its official evaluation uses an external judge. Use `per_sample.csv` with the official evaluator. Official leaderboard scripts remain authoritative for every benchmark.

## Reproducibility Checks

The allocator tests do not require model weights or benchmark data:

```bash
python -m unittest discover -s tests -v
```

Before publishing a release or archive, scan the repository for common identity and machine-specific strings:

```bash
python scripts/check_anonymity.py
```

The released model and budget configuration is recorded in [`configs/qwen2_5_vl_7b.json`](configs/qwen2_5_vl_7b.json).

## Implementation Guarantees

- The selected visual sequence is an order-preserving subsequence of the original decoder-visible sequence.
- Original M-RoPE position IDs are gathered without renumbering.
- Hidden states, attention masks, position IDs, and the Layer-0 KV cache are pruned consistently.
- Text and special tokens are never removed.
- Regional allocations respect capacity constraints and sum exactly to `B`.
- Recursive local cells are deterministic, non-overlapping, and cover each coarse region exactly.

## Repository Structure

```text
S2Prune/
├── assets/
│   └── s2prune_framework.png    method overview
├── configs/
│   └── qwen2_5_vl_7b.json       released protocol
├── s2prune/
│   ├── allocation.py            spatial allocation and local ERC selection
│   ├── data.py                  dataset loaders and prompt formatting
│   ├── metrics.py               answer parsing and local metrics
│   ├── qwen.py                  Qwen input construction and Layer-0 pruning
│   └── vizwiz_eval.py           VizWiz/VQA-style local metric
├── scripts/
│   ├── check_anonymity.py       release metadata scanner
│   ├── evaluate.py              benchmark evaluation entry point
│   └── smoke_test.py            one-image end-to-end check
├── tests/
│   └── test_allocation.py       deterministic allocator regression tests
├── pyproject.toml
└── requirements.txt
```

## Scope

This release targets Qwen2.5-VL-7B-Instruct with a fixed `672 × 672` image input and 576 decoder-visible visual tokens. It does not bundle model weights, benchmark datasets, cached images, generated predictions, or official leaderboard submission tools.
