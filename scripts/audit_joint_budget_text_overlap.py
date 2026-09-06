#!/usr/bin/env python
"""Audit paragraph-level reuse between the diagnostic and joint-budget drafts."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _paragraphs(text: str) -> list[str]:
    text = re.sub(r"(?m)^%.*$", "", text)
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"\\\[.*?\\\]", " ", text, flags=re.S)
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.S)
    text = re.sub(r"\\begin\{(?:equation|align|table|figure).*?\\end\{(?:equation|align|table|figure).*?\}", " ", text, flags=re.S)
    chunks = re.split(r"\n\s*\n", text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _tokens(paragraph: str) -> list[str]:
    paragraph = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^\]]*\])?", " ", paragraph)
    paragraph = re.sub(r"[#*_`{}$|<>\\]", " ", paragraph)
    return re.findall(r"[a-z]{3,}", paragraph.lower())


def _shingles(tokens: list[str], size: int) -> set[tuple[str, ...]]:
    if len(tokens) < size:
        return set()
    return {tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


def _jaccard(left: set[tuple[str, ...]], right: set[tuple[str, ...]]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _prepared(text: str, size: int = 5) -> list[dict]:
    prepared = []
    for index, paragraph in enumerate(_paragraphs(text)):
        tokens = _tokens(paragraph)
        if len(tokens) < 20:
            continue
        prepared.append(
            {
                "paragraph_index": index,
                "token_count": len(tokens),
                "tokens": tokens,
                "shingles": _shingles(tokens, size),
            }
        )
    return prepared


def _all_shingles(paragraphs: Iterable[dict], size: int) -> set[tuple[str, ...]]:
    result: set[tuple[str, ...]] = set()
    for paragraph in paragraphs:
        result.update(_shingles(paragraph["tokens"], size))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original",
        type=Path,
        default=ROOT / "paper" / "main_diagnostic.tex",
    )
    parser.add_argument(
        "--new-draft",
        type=Path,
        default=ROOT / "joint_group_label_budget" / "MAIN_CONFERENCE_DRAFT_EN.md",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "joint_group_label_budget"
        / "paper_assets"
        / "overlap_report.json",
    )
    args = parser.parse_args()

    original_text = args.original.read_text(encoding="utf-8")
    new_text = args.new_draft.read_text(encoding="utf-8")
    original = _prepared(original_text)
    new = _prepared(new_text)
    matches = []
    for new_paragraph in new:
        best = None
        for original_paragraph in original:
            score = _jaccard(
                new_paragraph["shingles"], original_paragraph["shingles"]
            )
            shared = len(
                new_paragraph["shingles"] & original_paragraph["shingles"]
            )
            candidate = (score, shared, original_paragraph)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        if best is not None:
            matches.append(
                {
                    "new_paragraph_index": new_paragraph["paragraph_index"],
                    "original_paragraph_index": best[2]["paragraph_index"],
                    "five_word_shingle_jaccard": best[0],
                    "shared_five_word_shingles": best[1],
                    "new_token_count": new_paragraph["token_count"],
                    "original_token_count": best[2]["token_count"],
                }
            )
    matches.sort(
        key=lambda record: (
            record["five_word_shingle_jaccard"],
            record["shared_five_word_shingles"],
        ),
        reverse=True,
    )
    original_long = _all_shingles(original, 12)
    new_long = _all_shingles(new, 12)
    shared_long = original_long & new_long
    maximum = matches[0]["five_word_shingle_jaccard"] if matches else 0.0
    flagged = [
        record
        for record in matches
        if record["five_word_shingle_jaccard"] >= 0.35
        and record["shared_five_word_shingles"] >= 5
    ]
    report = {
        "status": "pass" if not flagged and not shared_long else "review",
        "scope": (
            "Exact English paragraph reuse audit. Conceptual and cross-language "
            "scope overlap is assessed separately in ORIGINAL_PAPER_OVERLAP_CONTROL_ZH.md."
        ),
        "original": str(args.original),
        "original_sha256": _sha256(args.original),
        "new_draft": str(args.new_draft),
        "new_draft_sha256": _sha256(args.new_draft),
        "original_paragraphs_compared": len(original),
        "new_paragraphs_compared": len(new),
        "maximum_five_word_shingle_jaccard": maximum,
        "shared_twelve_word_shingle_count": len(shared_long),
        "flagged_paragraph_pairs": flagged,
        "top_matches": matches[:10],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
