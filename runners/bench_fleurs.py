#!/usr/bin/env python3
"""FLEURS multi-language benchmark runner.

Usage:
    python runners/bench_fleurs.py --engines glm fun --per-lang 100
    python runners/bench_fleurs.py --per-lang full          # full official test split

FLEURS is fetched automatically via huggingface_hub (cached under HF_HOME),
no manual download required.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from bench.models import get_engine
from bench.metrics import error_rate

FLEURS_LANGS = {
    "en_us":       {"metric": "wer", "train_src": "librispeech+gigaspeech"},
    "cmn_hans_cn": {"metric": "cer", "train_src": "aishell1+wenetspeech+magicdata"},
    "ko_kr":       {"metric": "cer", "train_src": "ksponspeech"},
    "ja_jp":       {"metric": "cer", "train_src": "cv_ja"},
    "yue_hant_hk": {"metric": "cer", "train_src": "cv_yue"},
    "de_de":       {"metric": "wer", "train_src": "mls_german"},
    "fr_fr":       {"metric": "wer", "train_src": "mls_french"},
    "es_419":      {"metric": "wer", "train_src": "mls_spanish"},
    "it_it":       {"metric": "wer", "train_src": "mls_italian"},
    "nl_nl":       {"metric": "wer", "train_src": "mls_dutch"},
    "pl_pl":       {"metric": "wer", "train_src": "mls_polish"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="glm,fun", help="comma list: glm,fun,qwen")
    ap.add_argument("--langs", default=None, help="comma list subset of FLEURS_LANGS keys")
    ap.add_argument("--per-lang", default="100", help="samples per language, or 'full'")
    ap.add_argument("--gpu", action="store_true", default=True)
    ap.add_argument("--cpu", dest="gpu", action="store_false")
    ap.add_argument("--fp32", action="store_true", help="use fp32 CTC (glm only)")
    ap.add_argument("--out", default="results/fleurs_results.json")
    args = ap.parse_args()
    run(args)


def run(args):
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    import soundfile as sf

    engine_names = [e.strip() for e in args.engines.split(",")]
    engines = {}
    for name in engine_names:
        if name == "glm":
            engines[name] = get_engine(name, use_gpu=args.gpu, quantized=not args.fp32)
        else:
            engines[name] = get_engine(name, use_gpu=args.gpu)

    lang_keys = args.langs.split(",") if args.langs else list(FLEURS_LANGS)
    engine_labels = {
        n: ({"ctc": "GLM-ASR-CTC-Final134k", "quantized": not args.fp32} if n == "glm"
            else {"ctc": "Qwen3-ASR-CTC-56916", "quantized": True} if n == "qwen"
            else {"ctc": "Qwen3-ASR-CTC-r2-142295", "quantized": True} if n == "qwen_r2"
            else {"ctc": "Qwen3-ASR-CTC-r2-fp16", "quantized": False} if n == "qwen_r2_fp16"
            else "fun-asr-nano-int4")
        for n in engine_names
    }
    meta = {
        "dataset": "google/fleurs official test split",
        "per_lang": args.per_lang,
        "engines": engine_labels,
        "onnxruntime": ort.__version__,
        "providers": {n: (engines[n].enc_sess.get_providers()[0] if hasattr(engines[n], "enc_sess") else "?") for n in engine_names},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    results = {}
    for lang in lang_keys:
        cfg = FLEURS_LANGS[lang]
        pq_path = hf_hub_download(
            "google/fleurs", f"parquet-data/{lang}/test-00000-of-00001.parquet", repo_type="dataset"
        )
        table = pq.read_table(pq_path)
        per_lang = table.num_rows if args.per_lang == "full" else int(args.per_lang)
        rows = table.to_pylist()
        step = max(1, table.num_rows // per_lang)
        rows = rows[::step][:per_lang]

        stats = {name: [0, 0] for name in engine_names}
        t_lang = time.time()
        for sample in rows:
            audio, sr = sf.read(io.BytesIO(sample["audio"]["bytes"]), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            ref = sample["transcription"]
            for name, engine in engines.items():
                hyp = engine.decode_text(engine.encode(audio))
                err, total = error_rate(ref, hyp, cfg["metric"], lang)
                stats[name][0] += err
                stats[name][1] += total

        entry = {"metric": cfg["metric"].upper(), "train_src": cfg["train_src"], "n": len(rows)}
        scores = {}
        for name in engine_names:
            er = stats[name][0] / max(stats[name][1], 1) * 100
            entry[f"{name}_er"] = round(er, 1)
            entry[f"{name}_err/total"] = stats[name]
            scores[name] = er
        results[lang] = entry
        winner = min(scores, key=scores.get)
        score_str = "  ".join(f"{n}={scores[n]:.1f}%" for n in engine_names)
        print(f"[{lang}] {cfg['metric'].upper()}  {score_str}  ({len(rows)} samples, {time.time()-t_lang:.0f}s) -> {winner}")

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"meta": meta, "results": results}, f, ensure_ascii=False, indent=2)

    print(f"\nresults -> {args.out}")


if __name__ == "__main__":
    main()
