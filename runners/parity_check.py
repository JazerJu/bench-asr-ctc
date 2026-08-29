#!/usr/bin/env python3
"""Provider-parity check: does each engine's transcript change CPU vs CUDA EP?

Uses the same deterministic strided sampling as bench_fleurs (--per-lang 200).
Exit verdict: identical transcripts => published mixed-provider numbers stand.
"""

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

from bench.models import get_engine
from bench.metrics import error_rate

LANGS = {"en_us": "wer", "cmn_hans_cn": "cer"}
PER_LANG = 200


def rows_of(lang):
    pq_path = hf_hub_download("google/fleurs", f"parquet-data/{lang}/test-00000-of-00001.parquet", repo_type="dataset")
    table = pq.read_table(pq_path)
    rows = table.to_pylist()
    step = max(1, table.num_rows // PER_LANG)
    return rows[::step][:PER_LANG]


def run_config(engine_name, use_gpu, batches):
    eng = get_engine(engine_name, use_gpu=use_gpu)
    hyps = []
    for rows in batches:
        for sample in rows:
            audio, sr = sf.read(io.BytesIO(sample["audio"]["bytes"]), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            hyps.append(eng.decode_text(eng.encode(audio)))
    return hyps


def main():
    batches = [rows_of(lang) for lang in LANGS]
    refs = [[s["transcription"] for s in rows] for rows in batches]

    for name in ("fun", "glm"):
        cpu = run_config(name, False, batches)
        gpu = run_config(name, True, batches)
        for i, lang in enumerate(LANGS):
            lo, hi = i * PER_LANG, (i + 1) * PER_LANG
            c, g = cpu[lo:hi], gpu[lo:hi]
            same = sum(1 for a, b in zip(c, g) if a == b)
            metric = LANGS[lang]
            er_c = sum(error_rate(r, h, metric, lang)[0] for r, h in zip(refs[i], c))
            er_g = sum(error_rate(r, h, metric, lang)[0] for r, h in zip(refs[i], g))
            tot = sum(error_rate(r, h, metric, lang)[1] for r, h in zip(refs[i], c))
            diff_idx = [j for j, (a, b) in enumerate(zip(c, g)) if a != b]
            print(f"[{name}/{lang}] transcript identical: {same}/{PER_LANG}  "
                  f"ER cpu={er_c/tot*100:.2f}% cuda={er_g/tot*100:.2f}%  "
                  f"diff@{diff_idx[:3]}")
            for j in diff_idx[:2]:
                print(f"    cpu: {c[j][:60]}")
                print(f"    gpu: {g[j][:60]}")


if __name__ == "__main__":
    main()
