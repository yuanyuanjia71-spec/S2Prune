#!/usr/bin/env python3
"""Evaluate S2Prune with the released Qwen2.5-VL protocol."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2prune.allocation import DEFAULT_GRIDS, default_grid_size
from s2prune.data import collect_samples, default_instruction, format_prompt
from s2prune.metrics import pope_statistics, score_sample
from s2prune.qwen import greedy_decode, load_model, prepare_input, s2prune_prefill


DATASETS = [
    "VQAv2", "TextVQA", "VizWiz", "MMBench", "ScienceQA",
    "POPE", "MME", "MMMU", "GQA", "MMVet",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument(
        "--manifest",
        default=None,
        help="Normalized JSON/JSONL manifest; required for MMMU, GQA, and MMVet.",
    )
    parser.add_argument("--split", default=None)
    parser.add_argument("--mmbench-lang", default="en", choices=["en", "cn", "cc"])
    parser.add_argument("--image-dir", default=None, help="Optional POPE image directory")
    parser.add_argument(
        "--budget", type=int, choices=sorted(DEFAULT_GRIDS), required=True
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=None,
        help="Defaults to 4, 5, 8, and 9 for B=32, 64, 128, and 192.",
    )
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def default_split(dataset: str) -> str:
    return {
        "VQAv2": "val",
        "TextVQA": "val",
        "VizWiz": "val",
        "MMBench": "dev",
        "ScienceQA": "test",
        "POPE": "adversarial",
        "MME": "test",
        "MMMU": "validation",
        "GQA": "testdev_balanced",
        "MMVet": "test",
    }[dataset]


def write_rows(path: Path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def mme_score(rows):
    grouped = {}
    for row in rows:
        parts = str(row["sample_id"]).split("/")
        key = "/".join(parts[:2]) if len(parts) >= 2 else str(row["image_path"])
        grouped.setdefault(key, []).append(float(row["score"]))
    categories = {}
    for key, scores in grouped.items():
        categories.setdefault(key.split("/", 1)[0].lower(), []).append(scores)
    total = 0.0
    for groups in categories.values():
        accuracy = np.mean([score for group in groups for score in group])
        accuracy_plus = np.mean([float(all(score >= 0.5 for score in group)) for group in groups])
        total += 100.0 * (accuracy + accuracy_plus)
    return float(total)


def main():
    args = parse_args()
    args.split = args.split or default_split(args.dataset)
    if not args.manifest and not args.data_root:
        raise ValueError("Provide --data-root or --manifest")
    grid_size = args.grid_size or default_grid_size(args.budget)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "image_cache"
    cache_dir.mkdir(exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    samples = collect_samples(args, cache_dir)
    if not samples:
        raise RuntimeError("No samples were loaded")

    model, processor, tokenizer = load_model(args.model_path, device=args.device)
    instruction = args.instruction or default_instruction(args.dataset)
    result_path = output_dir / "per_sample.csv"
    rows = []
    if result_path.exists():
        with result_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    completed = {str(row["sample_id"]) for row in rows}

    for index, sample in enumerate(samples, start=1):
        if str(sample["id"]) in completed:
            continue
        prompt = format_prompt(sample, args.dataset, instruction)
        image = Image.open(sample["image_path"]).convert("RGB")
        prepared = prepare_input(model, processor, prompt, image, target_visual_tokens=576)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        prefill = s2prune_prefill(
            model,
            prepared,
            budget=args.budget,
            grid_size=grid_size,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - start) * 1000.0
        raw_answer, generated_tokens = greedy_decode(
            model,
            tokenizer,
            prefill,
            prepared.eos_token_id,
            args.max_new_tokens,
            args.dataset,
        )
        final_answer, score = score_sample(raw_answer, sample, args.dataset)
        selected = prefill.selection.selected_indices.detach().cpu().tolist()
        row = {
            "sample_id": str(sample["id"]),
            "image_path": str(sample["image_path"]),
            "question": str(sample.get("question", "")),
            "gold": str(sample.get("answer", sample.get("answers", ""))),
            "raw_answer": raw_answer,
            "final_answer": final_answer,
            "score": float(score),
            "visual_tokens_before": 576,
            "visual_tokens_after": len(selected),
            "budget": args.budget,
            "coarse_grid_size": grid_size,
            "selected_indices": json.dumps(selected),
            "region_budgets": json.dumps(prefill.selection.region_budgets),
            "laplacian_complexity_raw": json.dumps(
                prefill.selection.raw_complexity.tolist()
            ),
            "laplacian_complexity_normalized": json.dumps(
                prefill.selection.normalized_complexity.tolist()
            ),
            "full_sequence_length": prefill.full_sequence_length,
            "pruned_sequence_length": prefill.pruned_sequence_length,
            "pruning_location": "after decoder Layer 0 / before Layer 1",
            "prefill_ms": prefill_ms,
            "generated_tokens": generated_tokens,
            "category": str(sample.get("category", "")),
        }
        if row["visual_tokens_after"] != args.budget:
            raise RuntimeError("The physical retained-token count does not equal B")
        rows.append(row)
        write_rows(result_path, rows)
        if index == 1 or index % 25 == 0:
            print(
                f"[{index}/{len(samples)}] id={sample['id']} "
                f"576->{len(selected)} score={score:.3f} answer={final_answer!r}",
                flush=True,
            )

    numeric_scores = [float(row["score"]) for row in rows]
    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "num_samples": len(rows),
        "method": "S2Prune",
        "model": args.model_path,
        "initial_visual_tokens": 576,
        "budget": args.budget,
        "coarse_grid_size": grid_size,
        "pruning_location": "after decoder Layer 0 / before Layer 1",
        "mean_sample_score": float(np.mean(numeric_scores)),
        "accuracy_percent": float(np.mean(numeric_scores) * 100.0),
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
    }
    if args.dataset == "POPE":
        summary.update(pope_statistics(rows))
    if args.dataset == "MME":
        summary["mme_total"] = mme_score(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
