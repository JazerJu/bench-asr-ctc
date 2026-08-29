#!/usr/bin/env python3
"""Hotword DP-matching benchmark: pypinyin phoneme + fuzzy edit-distance match.

Validates the CapsWriter-style hotword pipeline on the bundled case audio:
CTC first-pass text (never modified) -> PhonemeCorrector DP match -> hotword list.

Usage:
    python runners/bench_hotword.py --engines glm fun
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))

import numpy as np
import soundfile as sf

from bench.models import get_engine

ROOT = Path(__file__).resolve().parent.parent
CASES_DIR = ROOT / "cases"
HOTWORDS_PATH = ROOT / "hot.txt"

from bench.hotword.hot_phoneme import PhonemeCorrector


def load_hotwords(path: Path):
    words = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if item and not item.startswith("#"):
            words.append(item)
    return words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="glm,fun")
    ap.add_argument("--gpu", action="store_true", default=True)
    ap.add_argument("--cpu", dest="gpu", action="store_false")
    args = ap.parse_args()

    hotwords = load_hotwords(HOTWORDS_PATH)
    corrector = PhonemeCorrector(threshold=0.85, similar_threshold=0.6)
    corrector.update_hotwords(hotwords)

    engines = {}
    for name in [e.strip() for e in args.engines.split(",")]:
        kwargs = {"use_gpu": args.gpu} if name == "glm" else {}
        engines[name] = get_engine(name, **kwargs)

    cases = json.loads((CASES_DIR / "cases.json").read_text(encoding="utf-8"))["cases"]
    for case in cases:
        audio_path = CASES_DIR / case["audio"]
        audio, sr = sf.read(audio_path, dtype="float32")
        if sr != 16000:
            raise RuntimeError(f"{audio_path}: expected 16k, got {sr}")
        print(f"\n=== {case['name']} (expected hotwords: {case['expected_hotwords']}) ===")
        for name, engine in engines.items():
            text = engine.decode_text(engine.encode(audio))
            result = corrector.correct(text, k=10) if text else None
            matched = sorted({hw for _, hw, _ in (result.matchs + result.similars)}) if result else []
            hit = sorted(set(matched) & set(case["expected_hotwords"]))
            miss = sorted(set(case["expected_hotwords"]) - set(matched))
            fp = sorted(set(matched) - set(case["expected_hotwords"]))
            print(f"  [{name}] ctc: {text[:70]}...")
            print(f"  [{name}] matched={matched}  HIT={hit}  MISS={miss}  FALSE-POS={fp}")


if __name__ == "__main__":
    main()
