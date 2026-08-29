#!/usr/bin/env python3
"""bench-asr-ctc: GLM-ASR-CTC vs Fun-ASR-Nano (vs future Qwen3-ASR-CTC).

Reproducible FLEURS benchmark. Sampling is deterministic (evenly strided over
the official test split), so the same --counts always evaluates the same
sentences on any machine.

Quick sanity (fast):
    python bench.py --counts 20 --langs en_us,cmn_hans_cn

Issue-table scale (what backs the published numbers):
    python bench.py --counts 200

Full official test split (7,876 samples, ~1-2 h):
    python bench.py --counts full

Prereqs: python runners/../scripts/download_models.py fetch   (models from HF)
         unset proxies; HF_HOME defaults to ~/.cache/huggingface
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--counts", default="100", help="sentences per language, or 'full'")
    ap.add_argument("--engines", default="glm,fun", help="comma list: glm,fun,qwen")
    ap.add_argument("--langs", default=None, help="comma list, e.g. en_us,cmn_hans_cn,ja_jp")
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    ap.add_argument("--fp32", action="store_true", help="use fp32 GLM CTC head instead of int4")
    ap.add_argument("--provider", default="auto", help="auto|cuda|dml|cpu (Windows iGPU: dml)")
    ap.add_argument("--precision", default=None, help="q4|fp16|fp32 (Windows DML: fp16)")
    ap.add_argument("--out", default=None, help="output json path (default results/fleurs_c<counts>.json)")
    args = ap.parse_args()
    import os
    os.environ["BENCH_ORT_PROVIDER"] = args.provider
    if args.precision:
        os.environ["BENCH_PRECISION"] = args.precision

    counts = args.counts if args.counts == "full" else str(int(args.counts))
    out = args.out or f"results/fleurs_c{counts}.json"

    cmd = [sys.executable, str(ROOT / "runners" / "bench_fleurs.py"),
           "--engines", args.engines, "--per-lang", counts, "--out", out]
    if args.langs:
        cmd += ["--langs", args.langs]
    if args.cpu:
        cmd += ["--cpu"]
    if args.fp32:
        cmd += ["--fp32"]

    print(f"[bench] {' '.join(cmd[1:])}")
    sys.exit(subprocess.run(cmd, cwd=ROOT).returncode)


if __name__ == "__main__":
    main()
