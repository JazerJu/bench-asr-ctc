#!/usr/bin/env python3
"""LibriSpeech test-clean WER for the qwen engine — parity check vs the
training-side PyTorch measurement (WER 6.93% full split)."""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import soundfile as sf

from bench.models import get_engine
from bench.metrics import error_rate

ROOT = Path("/data/datasets/librispeech/LibriSpeech/test-clean")


def transcripts():
    for t in sorted(ROOT.rglob("*.trans.txt")):
        for line in open(t, encoding="utf-8"):
            m = re.match(r"(\S+) (.+)", line.strip())
            if m:
                yield m.group(1), m.group(2)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    eng = get_engine("qwen", use_gpu=True)
    err = tot = 0
    count = 0
    for utt_id, ref in transcripts():
        flac = next(ROOT.rglob(utt_id + ".flac"))
        audio, _ = sf.read(flac, dtype="float32")
        hyp = eng.decode_text(eng.encode(audio))
        e, t = error_rate(ref, hyp, "wer", "en_us")
        err += e
        tot += t
        count += 1
        if count >= n:
            break
    print(f"n={count}  WER={err/tot*100:.2f}%  (training-side PyTorch: 6.93%)")


if __name__ == "__main__":
    main()
