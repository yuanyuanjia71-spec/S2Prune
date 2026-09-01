#!/usr/bin/env python3
"""Run one image through all released S2Prune budget configurations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2prune.allocation import DEFAULT_GRIDS
from s2prune.qwen import greedy_decode, load_model, prepare_input, s2prune_prefill


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", default="Describe the image briefly.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()

    model, processor, tokenizer = load_model(args.model_path, device=args.device)
    image = Image.open(args.image).convert("RGB")
    for budget, grid_size in DEFAULT_GRIDS.items():
        prepared = prepare_input(model, processor, args.question, image)
        prefill = s2prune_prefill(model, prepared, budget, grid_size)
        selected = prefill.selection.selected_indices
        assert int(prepared.visual_tokens.shape[0]) == 576
        assert int(selected.numel()) == budget
        assert int(selected.unique().numel()) == budget
        assert selected.equal(selected.sort().values)
        assert sum(prefill.selection.region_budgets) == budget
        answer, generated_tokens = greedy_decode(
            model,
            tokenizer,
            prefill,
            prepared.eos_token_id,
            args.max_new_tokens,
            "MMVet",
        )
        assert generated_tokens > 0
        print(
            f"PASS B={budget} grid={grid_size}x{grid_size} "
            f"visual=576->{selected.numel()} "
            f"sequence={prefill.full_sequence_length}->{prefill.pruned_sequence_length} "
            f"generated={generated_tokens} answer={answer!r}"
        )


if __name__ == "__main__":
    main()
