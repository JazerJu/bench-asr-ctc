#!/usr/bin/env python3
"""Buckeye conversational-English WER, symmetric provider config (both CUDA)."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import glob
import soundfile as sf

from bench.models import get_engine
from bench.metrics import error_rate



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2478)
    ap.add_argument("--engines", default="glm,fun,qwen")
    ap.add_argument("--out", default="results/buckeye_cuda.json")
    args = ap.parse_args()

    snap = glob.glob("/data/.cache/huggingface/hub/datasets--alexwengg--buckeye/snapshots/*/")[0]
    manifest = json.load(open(snap + "manifest.json"))
    rows = manifest["samples"][:args.n]
    engines = {n: get_engine(n, use_gpu=True) for n in args.engines.split(",")}
    stats = {n: [0, 0] for n in engines}
    skipped = 0
    used = 0
    for i, sample in enumerate(rows):
        audio, _ = sf.read(snap + sample["audio"], dtype="float32")
        if audio is None or getattr(audio, "size", 0) == 0:
            skipped += 1
            continue
        ref = sample["transcript"]
        for name, eng in engines.items():
            hyp = eng.decode_text(eng.encode(audio))
            err, total = error_rate(ref, hyp, "wer", "en_us")
            stats[name][0] += err
            stats[name][1] += total
        used += 1
        if used % 200 == 0:
            print(f"  {used}/{len(rows)} skipped={skipped}", flush=True)

    out = {}
    for name, (err, total) in stats.items():
        out[name] = round(err / total * 100, 1)
        print(f"[{name}] WER={out[name]}%  ({err}/{total})")
    Path(args.out).write_text(json.dumps(
        {"n": used, "skipped_empty": skipped, "meta": {"providers": "CUDA"}, "wer": out}, indent=1))


if __name__ == "__main__":
    main()
