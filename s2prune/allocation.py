"""Spatial budget allocation and local representative selection for S2Prune.

This module is intentionally model-agnostic. It implements the exact
Laplacian allocation and recursive-cell ERC selection used in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image


Region = Tuple[int, int, int, int]
FP32_EPS = float(np.finfo(np.float32).eps)
FP64_EPS = float(np.finfo(np.float64).eps)
DEFAULT_GRIDS = {32: 4, 64: 5, 128: 8, 192: 9}


@dataclass(frozen=True)
class SelectionResult:
    """Complete allocation trace for one image."""

    selected_indices: torch.Tensor
    regions: List[Region]
    region_budgets: List[int]
    final_cells: List[Region]
    raw_complexity: np.ndarray
    normalized_complexity: np.ndarray


def default_grid_size(budget: int) -> int:
    """Return the paper configuration for a supported visual-token budget."""

    try:
        return DEFAULT_GRIDS[int(budget)]
    except KeyError as exc:
        raise ValueError(
            f"No paper-default grid is defined for B={budget}; pass grid_size explicitly."
        ) from exc


def build_coarse_regions(
    grid_h: int,
    grid_w: int,
    rows: int,
    cols: int,
) -> List[Region]:
    """Partition a token grid into near-equal non-overlapping rectangles.

    Boundaries use Python's ``round(i * size / parts)`` exactly as in the
    experimental implementation.
    """

    row_edges = [round(i * int(grid_h) / int(rows)) for i in range(int(rows) + 1)]
    col_edges = [round(i * int(grid_w) / int(cols)) for i in range(int(cols) + 1)]
    regions: List[Region] = []
    for row in range(int(rows)):
        for col in range(int(cols)):
            r0, r1 = row_edges[row], row_edges[row + 1]
            c0, c1 = col_edges[col], col_edges[col + 1]
            if r1 <= r0 or c1 <= c0:
                continue
            regions.append((r0, r1, c0, c1))
    return regions


def laplacian_region_complexity(
    image: Image.Image,
    regions: Sequence[Region],
    grid_h: int,
    grid_w: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute raw and per-image min-max normalized Laplacian variance.

    The image is converted to grayscale, padded by edge replication, and
    filtered with the four-neighbor discrete Laplacian kernel.
    """

    gray = np.asarray(image.convert("L"), dtype=np.float32)
    if gray.ndim != 2 or min(gray.shape) < 3:
        raise ValueError(f"Cannot compute Laplacian variance from image shape {gray.shape}")

    padded = np.pad(gray, 1, mode="edge")
    laplacian = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4.0 * gray
    )
    height, width = gray.shape
    raw_values = []
    for r0, r1, c0, c1 in regions:
        y0, y1 = round(int(r0) * height / grid_h), round(int(r1) * height / grid_h)
        x0, x1 = round(int(c0) * width / grid_w), round(int(c1) * width / grid_w)
        crop = laplacian[y0:y1, x0:x1]
        if crop.size == 0:
            raise RuntimeError(f"Empty Laplacian crop for region {(r0, r1, c0, c1)}")
        raw_values.append(float(np.var(crop, dtype=np.float64)))

    raw = np.asarray(raw_values, dtype=np.float32)
    denominator = max(float(raw.max() - raw.min()), FP32_EPS)
    normalized = (raw - raw.min()) / denominator
    return raw, normalized


def largest_remainder_allocation(
    region_scores: Sequence[float],
    region_sizes: Sequence[int],
    budget: int,
    minimum: int = 1,
) -> List[int]:
    """Allocate an exact budget using capped largest-fractional remainders."""

    scores = np.asarray(region_scores, dtype=np.float64).reshape(-1)
    capacities = np.asarray(region_sizes, dtype=np.int64).reshape(-1)
    if scores.size == 0 or scores.size != capacities.size:
        raise ValueError("region_scores and region_sizes must be non-empty and aligned")
    if not np.isfinite(scores).all() or np.any(scores < 0.0):
        raise ValueError("region_scores must be finite and non-negative")
    if np.any(capacities < int(minimum)):
        raise ValueError("Every region must have capacity >= minimum")
    if budget < minimum * scores.size or budget > int(capacities.sum()):
        raise ValueError(
            f"budget={budget} is infeasible for {scores.size} regions with minimum={minimum}"
        )

    allocated = np.full(scores.size, int(minimum), dtype=np.int64)
    remaining = int(budget - allocated.sum())
    while remaining:
        available = capacities - allocated
        eligible = np.flatnonzero(available > 0)
        eligible_before = eligible.copy()
        weights = scores[eligible].copy()
        if float(weights.sum()) <= FP64_EPS:
            weights.fill(1.0)

        quotas = weights / weights.sum() * remaining
        additions = np.minimum(np.floor(quotas).astype(np.int64), available[eligible])
        added = int(additions.sum())
        if added:
            allocated[eligible] += additions
            remaining -= added
            if remaining == 0:
                break

        available = capacities - allocated
        eligible = np.flatnonzero(available > 0)
        fractions = quotas - np.floor(quotas)
        fraction_by_region = {
            int(region): float(fraction)
            for region, fraction in zip(eligible_before, fractions)
        }
        order = sorted(
            eligible.tolist(),
            key=lambda idx: (fraction_by_region.get(int(idx), 0.0), scores[idx], -idx),
            reverse=True,
        )
        for region_idx in order:
            if remaining == 0:
                break
            if allocated[region_idx] < capacities[region_idx]:
                allocated[region_idx] += 1
                remaining -= 1

    if int(allocated.sum()) != int(budget):
        raise RuntimeError("Largest-remainder allocation did not preserve the budget")
    return allocated.tolist()


def recursive_region_cells(region: Region, budget: int) -> List[Region]:
    """Split a region into exactly ``budget`` deterministic rectangular cells.

    The largest splittable rectangle is bisected along its longer dimension.
    Rows win dimension ties, and integer floor midpoints produce floor/ceil
    children when a side length is odd.
    """

    r0, r1, c0, c1 = (int(value) for value in region)
    target = int(budget)
    capacity = (r1 - r0) * (c1 - c0)
    if target < 1 or target > capacity:
        raise ValueError(f"Cannot form {target} non-empty cells for region={region}")

    cells: List[Region] = [(r0, r1, c0, c1)]
    while len(cells) < target:
        candidates = [
            (-(b - a) * (d - c), index, a, b, c, d)
            for index, (a, b, c, d) in enumerate(cells)
            if (b - a) > 1 or (d - c) > 1
        ]
        if not candidates:
            raise RuntimeError(f"Could not split region={region} into {target} cells")
        _negative_area, index, a, b, c, d = min(candidates)
        if (b - a) >= (d - c) and (b - a) > 1:
            midpoint = a + (b - a) // 2
            children = [(a, midpoint, c, d), (midpoint, b, c, d)]
        else:
            midpoint = c + (d - c) // 2
            children = [(a, b, c, midpoint), (a, b, midpoint, d)]
        cells[index : index + 1] = children
    return cells


def select_s2prune_tokens(
    erc_scores: torch.Tensor,
    image: Image.Image,
    grid_h: int,
    grid_w: int,
    budget: int,
    grid_size: int | None = None,
) -> SelectionResult:
    """Run the complete S2Prune allocation and local ERC selection."""

    scores = erc_scores.reshape(-1)
    expected_tokens = int(grid_h) * int(grid_w)
    if int(scores.numel()) != expected_tokens:
        raise ValueError(
            f"ERC score count {scores.numel()} does not match grid {grid_h}x{grid_w}"
        )
    if grid_size is None:
        grid_size = default_grid_size(int(budget))
    grid_size = int(grid_size)
    if grid_size < 1 or grid_size > min(int(grid_h), int(grid_w)):
        raise ValueError(f"Invalid grid_size={grid_size} for {grid_h}x{grid_w}")

    regions = build_coarse_regions(grid_h, grid_w, grid_size, grid_size)
    if len(regions) != grid_size * grid_size:
        raise RuntimeError("Coarse grid contains an empty region")
    if int(budget) < len(regions):
        raise ValueError(f"B={budget} must cover all {len(regions)} coarse regions")

    raw, normalized = laplacian_region_complexity(image, regions, grid_h, grid_w)
    region_sizes = [
        (int(r1) - int(r0)) * (int(c1) - int(c0))
        for r0, r1, c0, c1 in regions
    ]
    allocation = largest_remainder_allocation(
        normalized, region_sizes, int(budget), minimum=1
    )

    selected: List[int] = []
    final_cells: List[Region] = []
    for region, region_budget in zip(regions, allocation):
        cells = recursive_region_cells(region, int(region_budget))
        final_cells.extend(cells)
        for r0, r1, c0, c1 in cells:
            candidates = torch.tensor(
                [row * grid_w + col for row in range(r0, r1) for col in range(c0, c1)],
                device=scores.device,
                dtype=torch.long,
            )
            choice = int(torch.argmax(scores.index_select(0, candidates)).item())
            selected.append(int(candidates[choice].item()))

    selected_indices = torch.tensor(
        sorted(selected), device=scores.device, dtype=torch.long
    )
    if int(selected_indices.numel()) != int(budget):
        raise RuntimeError(f"Selected {selected_indices.numel()} tokens, expected {budget}")
    if int(torch.unique(selected_indices).numel()) != int(budget):
        raise RuntimeError("S2Prune selected duplicate visual tokens")
    if not torch.equal(selected_indices, selected_indices.sort().values):
        raise RuntimeError("Selected visual tokens are not in original sequence order")
    if sum(allocation) != int(budget) or any(value < 1 for value in allocation):
        raise RuntimeError("Invalid regional budget allocation")

    return SelectionResult(
        selected_indices=selected_indices,
        regions=regions,
        region_budgets=allocation,
        final_cells=final_cells,
        raw_complexity=raw,
        normalized_complexity=normalized,
    )
