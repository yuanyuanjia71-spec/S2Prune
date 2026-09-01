"""Deterministic regression tests for the released S2Prune allocator."""

from __future__ import annotations

import hashlib
import unittest

import numpy as np
import torch
from PIL import Image

from s2prune.allocation import (
    build_coarse_regions,
    largest_remainder_allocation,
    recursive_region_cells,
    select_s2prune_tokens,
)


GOLDEN_SELECTED_HASHES = {
    32: "df8362790cd6c23585723b17d9cb48ff9566e3ac1311c12ac80fa1c0343672a5",
    64: "832405ce160618fee8987947ef93b408d2579db9cc17cb3530474c72d3f4171c",
    128: "889bbf89aaa06b2bb06b8a669b0ad031afe5cbd6b36c64be43d3f8fd372e9b5a",
}


class AllocationTest(unittest.TestCase):
    def test_coarse_regions_cover_grid_once(self):
        for grid_size in (4, 5, 8):
            coverage = np.zeros((24, 24), dtype=np.int64)
            regions = build_coarse_regions(24, 24, grid_size, grid_size)
            self.assertEqual(len(regions), grid_size**2)
            for r0, r1, c0, c1 in regions:
                coverage[r0:r1, c0:c1] += 1
            np.testing.assert_array_equal(coverage, np.ones_like(coverage))

    def test_recursive_cells_cover_region_once(self):
        region = (2, 9, 3, 11)
        for budget in (1, 2, 5, 13, 56):
            coverage = np.zeros((7, 8), dtype=np.int64)
            cells = recursive_region_cells(region, budget)
            self.assertEqual(len(cells), budget)
            for r0, r1, c0, c1 in cells:
                coverage[r0 - 2 : r1 - 2, c0 - 3 : c1 - 3] += 1
            np.testing.assert_array_equal(coverage, np.ones_like(coverage))

    def test_zero_scores_use_uniform_fallback(self):
        allocation = largest_remainder_allocation(
            [0.0, 0.0, 0.0, 0.0], [10, 10, 10, 10], budget=13
        )
        self.assertEqual(allocation, [4, 3, 3, 3])

    def test_golden_selection(self):
        generator = np.random.default_rng(2026)
        image = Image.fromarray(
            generator.integers(0, 256, size=(97, 113, 3), dtype=np.uint8),
            "RGB",
        )
        scores = torch.from_numpy(generator.normal(size=576).astype(np.float32))
        for budget, grid_size in ((32, 4), (64, 5), (128, 8)):
            result = select_s2prune_tokens(
                scores, image, 24, 24, budget, grid_size
            )
            digest = hashlib.sha256(
                result.selected_indices.cpu().numpy().tobytes()
            ).hexdigest()
            self.assertEqual(digest, GOLDEN_SELECTED_HASHES[budget])
            self.assertEqual(sum(result.region_budgets), budget)
            self.assertTrue(result.selected_indices.equal(result.selected_indices.sort().values))


if __name__ == "__main__":
    unittest.main()
