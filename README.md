<p align="center">
  <img src="assets/s2prune_logo.png" alt="S²Prune" width="240">
</p>

<h2 align="center">
  S²Prune: Spatially Structured Visual Token Pruning<br>
  for Multimodal Large Language Models
</h2>

<p align="center">
  Yuanyuan Jia &nbsp;·&nbsp; Shunpu Tang &nbsp;·&nbsp; Qianqian Yang<br>
  <sub>Zhejiang University</sub>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2609.01224"><strong>Paper</strong></a> &nbsp;·&nbsp;
  <a href="#quick-start"><strong>Quick start</strong></a> &nbsp;·&nbsp;
  <a href="docs/evaluation.md"><strong>Evaluation</strong></a> &nbsp;·&nbsp;
  <a href="#citation"><strong>Citation</strong></a>
</p>

<p align="center">
  <img src="assets/s2prune_framework.png" alt="S²Prune: spatial token-budget allocation followed by response-aware local sampling" width="100%">
  <br>
  <sub>Allocate tokens by spatial structure, then select informative local representatives.</sub>
</p>

**S²Prune** is a training-free method for reducing visual tokens in multimodal large language models. This release provides a fixed-budget implementation for **Qwen2.5-VL-7B-Instruct**, with physical sequence pruning after decoder Layer 0.

> **News:** S²Prune has been accepted to **WACV 2027**.

## Overview

- **Structure-aware allocation.** Assign more of the token budget to visually complex regions.
- **Response-aware sampling.** Retain one representative per local cell using early representation change.
- **Physical sequence pruning.** Shorten the sequence while preserving token order, M-RoPE coordinates, and all text and special tokens.

| Retained visual tokens | 32 | 64 | 128 | 192 |
| :--- | ---: | ---: | ---: | ---: |
| Share of the 576 input tokens | 5.6% | 11.1% | 22.2% | 33.3% |
| Coarse grid | 4 × 4 | 5 × 5 | 8 × 8 | 9 × 9 |

See the [method guide](docs/method.md) for the allocation rule, local selection score, and implementation guarantees.

## Quick start

### Installation

Tested with **Python 3.8.10**, **CUDA 12.1**, **PyTorch 2.4.1**, and **Transformers 4.49.0**. Dependency versions are pinned in [requirements.txt](requirements.txt).

```bash
git clone https://github.com/yuanyuanjia71-spec/S2Prune.git
cd S2Prune

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

### Run on one image

Use the Hugging Face model ID below or a local model snapshot. Model weights and benchmark data are downloaded separately.

```bash
export MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"

python scripts/smoke_test.py \
  --model-path "$MODEL_PATH" \
  --image /path/to/example.jpg \
  --device cuda:0
```

The smoke test runs all four released budgets and checks token counts, selected indices, and successful generation.

<details>
<summary>Expected token-count checks (abbreviated)</summary>

```text
PASS B=32 grid=4x4 visual=576->32 ...
PASS B=64 grid=5x5 visual=576->64 ...
PASS B=128 grid=8x8 visual=576->128 ...
PASS B=192 grid=9x9 visual=576->192 ...
```

</details>

## Evaluation

Run MMBench with a 64-token budget:

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

Each run writes `summary.json`, `per_sample.csv`, and `run_config.json`. The [evaluation guide](docs/evaluation.md) covers all ten benchmarks, JSON/JSONL manifests, scoring, and reproducibility checks.

## Documentation

| Guide | Contents |
| :--- | :--- |
| [Method and protocol](docs/method.md) | Spatial allocation, early-response selection, sequence pruning, and source map |
| [Evaluation](docs/evaluation.md) | Benchmarks, data interfaces, commands, outputs, and reproducibility |
| [Released configuration](configs/qwen2_5_vl_7b.json) | Model, image size, budgets, precision, and generation settings |

<details>
<summary>Run allocator checks without model weights</summary>

Run from the repository root:

```bash
python -m unittest discover -s tests -v
```

</details>

## Citation

If you find S²Prune useful, please cite our paper:

```bibtex
@misc{jia2026s2prune,
  title         = {S$^2$Prune: Spatially Structured Visual Token Pruning for Multimodal Large Language Models},
  author        = {Yuanyuan Jia and Shunpu Tang and Qianqian Yang},
  year          = {2026},
  eprint        = {2609.01224},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2609.01224}
}
```
