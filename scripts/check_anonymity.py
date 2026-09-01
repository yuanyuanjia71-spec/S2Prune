#!/usr/bin/env python3
"""Fail when common identity or machine-specific strings enter the release."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP = {Path(__file__).resolve()}
PATTERNS = {
    "absolute home path": re.compile(r"/(?:home|Users)/[^/\s]+/"),
    "email address": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "IPv4 address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "remote host": re.compile(r"(?:seetacloud|autodl|wandb)", re.IGNORECASE),
}


def main():
    findings = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.resolve() in SKIP or ".git" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            for label, pattern in PATTERNS.items():
                if pattern.search(line):
                    findings.append(f"{path.relative_to(ROOT)}:{line_number}: {label}")
    if findings:
        raise SystemExit("\n".join(findings))
    print("PASS: no identity or machine-specific strings detected")


if __name__ == "__main__":
    main()
