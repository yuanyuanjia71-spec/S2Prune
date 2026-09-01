"""Answer parsing and local benchmark metrics."""

from __future__ import annotations

import re
import string
from typing import Any, Dict, Iterable, List, Tuple

from .vizwiz_eval import official_vqa_accuracy


YES_NO_DATASETS = {"MME", "POPE"}


def extract_final_answer(text: str) -> str:
    value = str(text or "")
    boxed = re.search(
        r"Final\s*Answer\s*:\s*\\boxed\s*\{\s*(.*?)\s*\}",
        value,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if boxed:
        return " ".join(boxed.group(1).strip().split())
    final_line = re.search(r"Final\s*Answer\s*:\s*(.+)", value, flags=re.IGNORECASE)
    if final_line:
        return " ".join(final_line.group(1).rstrip(".").strip().split())
    return ""


def extract_choice(text: str) -> str:
    final = extract_final_answer(text)
    value = final or str(text)
    for pattern in (
        r"\\boxed\s*\{\s*([A-Z])\s*\}",
        r"Final\s*Answer\s*:\s*([A-Z])\b",
        r"^\s*([A-Z])\s*[.)]",
        r"(?:answer\s+is|answer)\s*[:：]?\s*([A-Z])\b",
    ):
        match = re.search(pattern, value, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    candidates = re.findall(r"\b([A-Z])\b", value, flags=re.IGNORECASE)
    return candidates[-1].upper() if candidates else ""


def extract_yes_no(text: str) -> str:
    final = extract_final_answer(text)
    value = final or str(text)
    candidates = re.findall(r"\b(yes|no)\b", value, flags=re.IGNORECASE)
    return candidates[-1].lower() if candidates else ""


def normalize_vqa_answer(text: str) -> str:
    value = str(text or "").lower().strip().replace("\n", " ").replace("\t", " ")
    value = value.translate(str.maketrans("", "", string.punctuation))
    value = " ".join(word for word in value.split() if word not in {"a", "an", "the"})
    return " ".join(value.split())


def vqa_soft_score(prediction: str, answers: Iterable[Any]) -> float:
    predicted = normalize_vqa_answer(prediction)
    references = []
    for item in answers:
        answer = item.get("answer", "") if isinstance(item, dict) else str(item)
        normalized = normalize_vqa_answer(answer)
        if normalized:
            references.append(normalized)
    if not predicted or not references:
        return 0.0
    return float(min(1.0, sum(answer == predicted for answer in references) / 3.0))


def score_sample(raw_answer: str, sample: Dict[str, Any], dataset: str) -> Tuple[str, float]:
    """Return the parsed answer and the local per-sample score."""

    if dataset in {"MMBench", "ScienceQA", "MMMU"} and sample.get("options"):
        answer = extract_choice(raw_answer)
        return answer, float(answer == str(sample.get("answer", "")).strip().upper())
    if dataset in YES_NO_DATASETS:
        answer = extract_yes_no(raw_answer)
        return answer, float(answer == str(sample.get("answer", "")).strip().lower())
    if dataset == "MMVet":
        return raw_answer.strip(), 0.0
    answer = extract_final_answer(raw_answer) or raw_answer.strip()
    if dataset == "VizWiz":
        return answer, official_vqa_accuracy(answer, sample.get("answers", []))
    if dataset == "GQA":
        references = sample.get("answers", [sample.get("answer", "")])
        gold = references[0] if references else ""
        return answer, float(normalize_vqa_answer(answer) == normalize_vqa_answer(gold))
    return answer, vqa_soft_score(answer, sample.get("answers", []))


def pope_statistics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    true_positive = sum(row["final_answer"] == "yes" and row["gold"] == "yes" for row in rows)
    false_positive = sum(row["final_answer"] == "yes" and row["gold"] == "no" for row in rows)
    false_negative = sum(row["final_answer"] == "no" and row["gold"] == "yes" for row in rows)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1}
