#!/usr/bin/env python3
"""Rebuild the a6_avg5/a7_avg5 regression table from saved JSON; no audio/models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NON_CHINESE = {"v669", "v623", "v736"}


def summarize(results: Path, label: str):
    def load(prefix):
        return json.loads((results / f"{prefix}_{label}.json").read_text(encoding="utf-8"))

    zh = [r for r in load("zh_long20")["rows"] if r["clip"] not in NON_CHINESE]
    en = load("en_regression")["rows"]
    parts = load("ted_collection")["parts"]
    if len(zh) != 17 or len(en) != 6 or len(parts) != 29:
        raise ValueError(f"unexpected corpus sizes for {label}: {len(zh)}, {len(en)}, {len(parts)}")
    return {
        "label": label,
        "zh_cer": sum(r["cer"] for r in zh) / len(zh),
        "zh_ratio": sum(r["ctc_ratio"] for r in zh) / len(zh),
        "en_cer": sum(r["cer"] for r in en) / len(en),
        "ted_ctc_han": sum(p["ctc_han"] for p in parts.values()),
        "ted_parts_with_han": sum(p["ctc_han"] > 0 for p in parts.values()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--out", type=Path, help="optional machine-readable summary")
    args = ap.parse_args()
    rows = [summarize(args.results, f"{round_}_{precision}")
            for round_ in ("a6avg5", "a7avg5")
            for precision in ("fp16", "q4f16", "hqqf16")]
    print("| Head | zh17 CER | zh17 character ratio | en6 CER | TED CTC Han | TED parts with Han |")
    print("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['label']} | {r['zh_cer']:.2%} | {r['zh_ratio']:.1%} | "
              f"{r['en_cer']:.2%} | {r['ted_ctc_han']} | {r['ted_parts_with_han']}/29 |")
    print("\nCER/ratio are unweighted means over clips, not a pooled corpus CER.")
    print("References are saved v1+LLM outputs, not manually verified transcripts.")
    print("TED overlaps TED-LIUM training; these are regression, not held-out, results.")
    if args.out:
        args.out.write_text(json.dumps({
            "aggregation": "macro mean over clips",
            "zh_excluded_non_chinese": sorted(NON_CHINESE),
            "reference": "saved v1+LLM outputs",
            "ted_training_overlap": True,
            "rows": rows,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
