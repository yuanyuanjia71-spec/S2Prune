"""S2Prune: training-free structure-aware visual-token pruning."""

from .allocation import (
    DEFAULT_GRIDS,
    SelectionResult,
    build_coarse_regions,
    default_grid_size,
    laplacian_region_complexity,
    largest_remainder_allocation,
    recursive_region_cells,
    select_s2prune_tokens,
)

__all__ = [
    "DEFAULT_GRIDS",
    "SelectionResult",
    "build_coarse_regions",
    "default_grid_size",
    "laplacian_region_complexity",
    "largest_remainder_allocation",
    "recursive_region_cells",
    "select_s2prune_tokens",
]
