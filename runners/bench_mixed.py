#!/usr/bin/env python3
"""Mixed zh/en sub-benchmark on FLEURS cmn_hans_cn + yue_hant_hk.

Reports three cuts per language:
  - pure   : Chinese-only utterances, CER (han chars only)
  - mixed  : zh/en-mixed utterances, CER (han chars only, en stripped)
  - mixed-CER: zh/en-mixed utterances, han chars as units + latin words as units
Usage: python runners/bench_mixed.py --engines glm,fun
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.models import get_engine

EN_WORD = re.compile(r"[a-zA-Z]{2,}")
HAN = re.compile(r"[\u4e00-\u9fff]")
LATIN_TOKEN = re.compile(r"[a-zA-Z]+")


def han_units(text: str):
    return re.findall(r"[\u4e00-\u9fff]", text)


def mixed_units(text: str):
    # han chars as single units + latin words (>=2 letters) as single units
    units = []
    buf = ""
    for ch in text.lower():
        if re.match(r"[a-z]", ch):
            buf += ch
            continue
        if buf:
            if len(buf) >= 2:
                units.append(buf)
            buf = ""
        if re.match(r"[\u4e00-\u9fff]", ch):
            units.append(ch)
    if len(buf) >= 2:
        units.append(buf)
    return units


def edit_distance_units(r, h):
    n, m = len(r), len(h)
    if n == 0:
        return m, 0
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]; dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (0 if r[i-1] == h[j-1] else 1))
            prev = cur
    return dp[m], n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="glm,fun,qwen")
    ap.add_argument("--gpu", action="store_true", default=True)
    ap.add_argument("--cpu", dest="gpu", action="store_false")
    ap.add_argument("--langs", default="cmn_hans_cn,yue_hant_hk")
    ap.add_argument("--out", default="results/fleurs_mixed.json")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    import soundfile as sf

    engines = {}
    for name in [e.strip() for e in args.engines.split(",")]:
        kwargs = {"use_gpu": args.gpu}
        engines[name] = get_engine(name, **kwargs)

    results = {}
    for lang in args.langs.split(","):
        pq_path = hf_hub_download("google/fleurs", f"parquet-data/{lang}/test-00000-of-00001.parquet", repo_type="dataset")
        rows = pq.read_table(pq_path).to_pylist()

        groups = {"pure": [], "mixed": []}
        for r in rows:
            tr = r["transcription"]
            (groups["mixed"] if (HAN.search(tr) and EN_WORD.search(tr)) else groups["pure"]).append(r)

        lang_res = {}
        for gname, grp in groups.items():
            stats = {}
            for name, engine in engines.items():
                han_e = han_t = mix_e = mix_t = 0
                t0 = time.time()
                for sample in grp:
                    audio, sr = sf.read(io.BytesIO(sample["audio"]["bytes"]), dtype="float32")
                    if audio.ndim > 1:
                        audio = audio.mean(axis=1)
                    hyp = engine.decode_text(engine.encode(audio))
                    ref_tr = sample["transcription"].lower()
                    # CER (han only)
                    e, t = edit_distance_units(han_units(ref_tr), han_units(hyp))
                    han_e += e; han_t += t
                    # mixed-CER (han chars + latin words)
                    e, t = edit_distance_units(mixed_units(ref_tr), mixed_units(hyp))
                    mix_e += e; mix_t += t
                stats[name] = {
                    "cer_han": round(han_e / max(han_t, 1) * 100, 1),
                    "mixed_cer": round(mix_e / max(mix_t, 1) * 100, 1),
                    "han_err/total": [han_e, han_t],
                    "mixed_err/total": [mix_e, mix_t],
                }
                stats[name]["sec"] = round(time.time() - t0)
            lang_res[gname] = {"n": len(grp), **stats}

        results[lang] = lang_res
        for gname, g in lang_res.items():
            line = "  ".join(f"{n}: CER={s['cer_han']}% mixed-CER={s['mixed_cer']}%" for n, s in g.items() if n != "n")
            print(f"[{lang}/{gname}] n={g['n']}  {line}")

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nresults -> {args.out}")


if __name__ == "__main__":
    main()
