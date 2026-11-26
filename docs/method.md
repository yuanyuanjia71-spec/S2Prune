# Method and released protocol

[← Back to README](../README.md)

The released implementation targets **Qwen2.5-VL-7B-Instruct** with a fixed `672 × 672` image input. Its PatchMerger produces a `24 × 24` grid containing **576** decoder-visible visual tokens.

Model weights, benchmark datasets, cached images, generated predictions, and official leaderboard submission tools are not bundled with this release.

## 1. Allocate tokens by structural density

For each coarse region `g`, S²Prune measures structural complexity with the variance of the image Laplacian:

$$
c_g = \operatorname{Var}(\operatorname{Laplacian}(I_g)).
$$

Every region receives one token first. The remaining budget is distributed according to the per-image min-max normalized complexity scores using capacity-aware largest-remainder allocation.

| Visual budget `B` | Coarse grid | Regions | Retained visual tokens |
| ---: | :---: | ---: | ---: |
| 32 | 4 × 4 | 16 | 5.6% |
| 64 | 5 × 5 | 25 | 11.1% |
| 128 | 8 × 8 | 64 | 22.2% |
| 192 | 9 × 9 | 81 | 33.3% |

## 2. Select one representative per local cell

Each region is recursively partitioned into exactly `B_g` non-overlapping cells. After the full visual sequence passes through decoder Layer 0, the early representation change (ERC) score for visual token `i` is

$$
s_i = \lVert h_i^1 - h_i^0 \rVert_2.
$$

The maximum-ERC token is retained in each cell. Selected indices are sorted into their original decoder-visible order, and physical sequence deletion is applied before decoder Layer 1.

## Implementation Guarantees

- The selected visual sequence is an order-preserving subsequence of the original decoder-visible sequence.
- Original M-RoPE position IDs are gathered without renumbering.
- Hidden states, attention masks, position IDs, and the Layer-0 KV cache are pruned consistently.
- Text and special tokens are never removed.
- Regional allocations respect capacity constraints and sum exactly to `B`.
- Recursive local cells are deterministic, non-overlapping, and cover each coarse region exactly.

## Released configuration

The exact model and budget configuration is recorded in [configs/qwen2_5_vl_7b.json](../configs/qwen2_5_vl_7b.json).

| Setting | Value |
| :--- | :--- |
| Backbone | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Image size | 672 × 672 |
| Original visual tokens | 576 |
| Pruning point | After decoder Layer 0 |
| Precision | `bfloat16` |
| Attention implementation | `eager` |
| Seed | 42 |

## Source map

| Component | Source |
| :--- | :--- |
| Spatial allocation and local token selection | [s2prune/allocation.py](../s2prune/allocation.py) |
| Qwen input construction, pruning, and decoding | [s2prune/qwen.py](../s2prune/qwen.py) |
| Benchmark loading and prompt formatting | [s2prune/data.py](../s2prune/data.py) |
| Answer normalization and scoring | [s2prune/metrics.py](../s2prune/metrics.py) |
| VizWiz scoring | [s2prune/vizwiz_eval.py](../s2prune/vizwiz_eval.py) |
| Evaluation entry point | [scripts/evaluate.py](../scripts/evaluate.py) |
