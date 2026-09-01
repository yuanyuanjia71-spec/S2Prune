"""Dataset loading and prompt formatting used by the released evaluator."""

from __future__ import annotations

import argparse
import json
import math
import random
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image


CHOICE_LABELS = ["A", "B", "C", "D", "E"]


def _slice(samples: List[Dict[str, Any]], offset: int, limit: int) -> List[Dict[str, Any]]:
    samples = samples[int(offset) :]
    return samples[: int(limit)] if int(limit) > 0 else samples


def collect_manifest(path: str, offset: int = 0, limit: int = 0) -> List[Dict[str, Any]]:
    """Load a JSON or JSONL manifest with normalized sample dictionaries."""

    manifest_path = Path(path)
    text = manifest_path.read_text(encoding="utf-8").strip()
    if not text:
        raise FileNotFoundError(f"Empty manifest: {manifest_path}")
    if text.startswith("["):
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    samples = []
    for index, row in enumerate(rows):
        sample = dict(row)
        sample.setdefault("id", str(index))
        image_path = Path(str(sample.get("image_path", "")))
        if not image_path.is_absolute():
            image_path = manifest_path.parent / image_path
        sample["image_path"] = str(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Manifest image does not exist: {image_path}")
        samples.append(sample)
    return _slice(samples, offset, limit)


def collect_mmbench(root: str, language: str, split: str, cache_dir: Path):
    import pandas as pd

    parquet_path = Path(root) / language / f"{split}-00000-of-00001.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"MMBench parquet not found: {parquet_path}")
    frame = pd.read_parquet(parquet_path)
    image_cache = cache_dir / f"mmbench_{language}_{split}"
    image_cache.mkdir(parents=True, exist_ok=True)
    samples = []
    for _, row in frame.iterrows():
        sample_id = str(row["index"])
        image_path = image_cache / f"{sample_id}.jpg"
        if not image_path.exists():
            payload = row["image"]
            image_bytes = payload.get("bytes") if isinstance(payload, dict) else payload
            if image_bytes is None:
                raise ValueError(f"Unsupported MMBench image for {sample_id}")
            Image.open(BytesIO(image_bytes)).convert("RGB").save(
                image_path, format="JPEG", quality=95
            )
        options = []
        for label in ["A", "B", "C", "D"]:
            value = row.get(label)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                break
            options.append((label, str(value)))
        hint = row.get("hint", "")
        if hint is None or (isinstance(hint, float) and math.isnan(hint)):
            hint = ""
        samples.append({
            "id": sample_id,
            "image_path": str(image_path),
            "question": str(row["question"]),
            "hint": str(hint).strip(),
            "options": options,
            "answer": str(row["answer"]).strip().upper(),
            "category": str(row.get("category", "")),
        })
    return samples


def collect_scienceqa(root: str, split: str):
    root_path = Path(root)
    problems = json.loads(
        (root_path / "data" / "scienceqa" / "problems.json").read_text(encoding="utf-8")
    )
    split_path = root_path / "data" / "scienceqa" / "pid_splits.json"
    if split_path.exists():
        question_ids = [str(value) for value in json.loads(
            split_path.read_text(encoding="utf-8")
        ).get(split, [])]
    else:
        question_ids = [qid for qid, row in problems.items() if row.get("split") == split]
    samples = []
    for question_id in question_ids:
        row = problems.get(str(question_id))
        if not row or not row.get("image") or not row.get("choices"):
            continue
        image_path = root_path / split / str(question_id) / str(row["image"])
        answer_index = row.get("answer")
        if not image_path.exists() or not isinstance(answer_index, int):
            continue
        samples.append({
            "id": str(question_id),
            "image_path": str(image_path),
            "question": str(row.get("question", "")),
            "hint": str(row.get("hint", "") or ""),
            "options": list(zip(CHOICE_LABELS, [str(x) for x in row["choices"]])),
            "answer": CHOICE_LABELS[answer_index],
            "category": str(row.get("category", "")),
        })
    return samples


def collect_textvqa(root: str):
    root_path = Path(root)
    annotation_path = root_path / "TextVQA_0.5.1_val.json"
    image_dir = root_path / "val_images" / "train_images"
    rows = json.loads(annotation_path.read_text(encoding="utf-8")).get("data", [])
    samples = []
    for row in rows:
        image_id = str(row.get("image_id", ""))
        image_path = image_dir / f"{image_id}.jpg"
        if image_path.exists():
            samples.append({
                "id": str(row.get("question_id", image_id)),
                "image_path": str(image_path),
                "question": str(row.get("question", "")),
                "answers": [str(x) for x in row.get("answers", []) if str(x).strip()],
                "category": "TextVQA",
            })
    return samples


def _normalize_yes_no(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"yes", "y", "true", "1"}:
        return "yes"
    if text in {"no", "n", "false", "0"}:
        return "no"
    return ""


def collect_pope(root: str, split: str, image_dir: str | None = None):
    root_path = Path(root)
    annotation_path = root_path / "coco" / f"coco_pope_{split}.json"
    resolved_image_dir = Path(image_dir) if image_dir else root_path / "val2014"
    text = annotation_path.read_text(encoding="utf-8").strip()
    rows = json.loads(text) if text.startswith("[") else [
        json.loads(line) for line in text.splitlines() if line.strip()
    ]
    samples = []
    for index, row in enumerate(rows):
        image_value = row.get("image", row.get("image_id", ""))
        if str(image_value).isdigit():
            image_path = resolved_image_dir / f"COCO_val2014_{int(image_value):012d}.jpg"
        else:
            image_path = resolved_image_dir / Path(str(image_value)).name
        answer = _normalize_yes_no(row.get("label", row.get("answer", "")))
        if image_path.exists() and answer:
            samples.append({
                "id": str(row.get("question_id", row.get("id", f"{split}:{index}"))),
                "image_path": str(image_path),
                "question": str(row.get("text", row.get("question", ""))),
                "answer": answer,
                "category": str(row.get("category", split)),
            })
    return samples


def collect_vizwiz(root: str, split: str):
    root_path = Path(root)
    rows = json.loads(
        (root_path / "annotations" / f"{split}.json").read_text(encoding="utf-8")
    )
    return [{
        "id": str(row["image"]),
        "image_path": str(root_path / split / str(row["image"])),
        "question": str(row["question"]),
        "answers": list(row.get("answers", [])),
        "category": "VizWiz",
    } for row in rows]


def _find_vqav2_files(root: Path, split: str):
    question_name = f"v2_OpenEnded_mscoco_{split}2014_questions.json"
    answer_name = f"v2_mscoco_{split}2014_annotations.json"
    question_path = root / "questions" / question_name
    answer_path = root / "annotations" / answer_name
    if not question_path.exists():
        question_path = next(root.rglob(question_name))
    if not answer_path.exists():
        answer_path = next(root.rglob(answer_name))
    image_dir = root / "images" / f"{split}2014"
    return question_path, answer_path, image_dir


def collect_vqav2(root: str, split: str, seed: int):
    root_path = Path(root)
    question_path, answer_path, image_dir = _find_vqav2_files(root_path, split)
    question_rows = json.loads(question_path.read_text(encoding="utf-8")).get("questions", [])
    answer_rows = json.loads(answer_path.read_text(encoding="utf-8")).get("annotations", [])
    questions = {int(row["question_id"]): str(row["question"]) for row in question_rows}
    answer_rows = [row for row in answer_rows if int(row["question_id"]) in questions]
    random.Random(int(seed)).shuffle(answer_rows)
    samples = []
    for row in answer_rows:
        question_id = int(row["question_id"])
        image_id = int(row["image_id"])
        image_name = f"COCO_{split}2014_{image_id:012d}.jpg"
        candidates = [image_dir / image_name, root_path / f"{split}2014" / image_name]
        image_path = next((path for path in candidates if path.exists()), None)
        if image_path is None:
            matches = list(root_path.rglob(image_name))
            image_path = matches[0] if matches else None
        if image_path is not None:
            samples.append({
                "id": str(question_id),
                "image_path": str(image_path),
                "question": questions[question_id],
                "answers": [
                    str(item.get("answer", "")).strip()
                    for item in row.get("answers", [])
                    if item.get("answer") is not None
                ],
            })
    return samples


def collect_mme(root: str):
    root_path = Path(root)
    samples = []
    for text_path in sorted(root_path.glob("*/*.txt")):
        image_path = next(
            (text_path.with_suffix(ext) for ext in (".jpg", ".jpeg", ".png", ".bmp")
             if text_path.with_suffix(ext).exists()),
            None,
        )
        if image_path is None:
            continue
        for line_index, line in enumerate(text_path.read_text(encoding="utf-8").splitlines()):
            parts = line.strip().split("\t")
            if len(parts) >= 2 and _normalize_yes_no(parts[-1]):
                samples.append({
                    "id": f"{text_path.parent.name}/{text_path.stem}/{line_index}",
                    "image_path": str(image_path),
                    "question": parts[0].strip(),
                    "answer": _normalize_yes_no(parts[-1]),
                    "category": text_path.parent.name,
                })
    return samples


def collect_samples(args: argparse.Namespace, cache_dir: Path):
    """Dispatch to a native loader or an explicit normalized manifest."""

    if args.manifest:
        return collect_manifest(args.manifest, args.sample_offset, args.max_images)
    if args.dataset == "MMBench":
        samples = collect_mmbench(args.data_root, args.mmbench_lang, args.split, cache_dir)
    elif args.dataset == "ScienceQA":
        samples = collect_scienceqa(args.data_root, args.split)
    elif args.dataset == "TextVQA":
        samples = collect_textvqa(args.data_root)
    elif args.dataset == "POPE":
        samples = collect_pope(args.data_root, args.split, args.image_dir)
    elif args.dataset == "VizWiz":
        samples = collect_vizwiz(args.data_root, args.split)
    elif args.dataset == "VQAv2":
        samples = collect_vqav2(args.data_root, args.split, args.seed)
    elif args.dataset == "MME":
        samples = collect_mme(args.data_root)
    else:
        raise ValueError(f"{args.dataset} requires --manifest in the anonymous release")
    return _slice(samples, args.sample_offset, args.max_images)


def default_instruction(dataset: str) -> str:
    if dataset == "ScienceQA":
        return "Answer with the option letter only (A, B, C, D, or E)."
    if dataset in {"MMBench", "MMMU"}:
        return "Answer with the option letter only (A, B, C, or D)."
    if dataset in {"POPE", "MME"}:
        return "Answer yes or no only."
    return "Answer with a short answer only."


def format_prompt(sample: Dict[str, Any], dataset: str, instruction: str) -> str:
    """Apply the same answer-only prompts as the experiments."""

    parts = ["<image>"]
    hint = str(sample.get("hint", "")).strip()
    if hint:
        parts.append(f"{'Context' if dataset == 'ScienceQA' else 'Hint'}: {hint}")
    parts.append(f"Question: {sample['question']}")
    for label, option in sample.get("options", []):
        parts.append(f"{label}. {option}")
    if instruction.strip():
        parts.append(instruction.strip())
    parts.append("Answer:")
    return "\n".join(parts)
