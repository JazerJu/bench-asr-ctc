#!/usr/bin/env python3
"""Paired zh/en CTC comparison on the SGLang tech talk (66min, no SRT).

No external GT: cuts 20s windows, runs BOTH engines, extracts English tokens,
and reports agreement. Disagreement windows get adjudicated separately by the
full-pipeline LLM decoder (cross-check runner).

Usage: python runners/bench_sglang.py [--minutes 20] [--seg-len 20]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.models import get_engine

ROOT = Path(__file__).resolve().parent.parent
VIDEO = Path("/data/其他模型/小型语言模型/sglang-dsv4.mp4")


def extract_terms(text: str) -> set[str]:
    """English tokens: lowercase runs of >=2 letters, after merging spaced
    single-letter sequences like 'c u d a' -> 'cuda'."""
    t = text.lower()
    t = re.sub(r"\b([a-z])\s+(?=[a-z]\b)", r"\1", t)
    return set(re.findall(r"[a-z][a-z0-9]{1,}", t))


def cut_wav(seg_start, seg_end, tmp):
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{seg_start:.2f}", "-to", f"{seg_end:.2f}",
         "-i", str(VIDEO), "-ar", "16000", "-ac", "1", str(tmp)],
        check=True, capture_output=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--seg-len", type=float, default=20.0)
    ap.add_argument("--out", default="results/sglang_pairs.json")
    args = ap.parse_args()

    import soundfile as sf

    dur = args.minutes * 60
    n = int(dur // args.seg_len)
    glm = get_engine("glm", use_gpu=True)
    fun = get_engine("fun")

    pairs = []
    for i in range(n):
        s, e = i * args.seg_len, (i + 1) * args.seg_len
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "seg.wav"
            cut_wav(s, e, wav)
            audio, _ = sf.read(wav, dtype="float32")
            g = glm.decode_text(glm.encode(audio))
            f = fun.decode_text(fun.encode(audio))
        gt, ft = extract_terms(g), extract_terms(f)
        pairs.append({
            "t": s, "glm": g, "fun": f,
            "glm_terms": sorted(gt), "fun_terms": sorted(ft),
            "both": sorted(gt & ft), "glm_only": sorted(gt - ft), "fun_only": sorted(ft - gt),
        })
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n}", flush=True)

    both = set().union(*[set(p["both"]) for p in pairs]) if pairs else set()
    go = [t for p in pairs for t in p["glm_only"]]
    fo = [t for p in pairs for t in p["fun_only"]]
    from collections import Counter
    summary = {
        "n_windows": n,
        "agreed_terms": len(both),
        "glm_only_total": len(go), "fun_only_total": len(fo),
        "glm_only_top": Counter(go).most_common(15),
        "fun_only_top": Counter(fo).most_common(15),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    Path(args.out).write_text(json.dumps({"summary": summary, "pairs": pairs}, ensure_ascii=False, indent=1))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
