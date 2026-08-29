#!/usr/bin/env python3
"""Spontaneous zh/en code-switch benchmark from a real tech vlog with word-level SRT.

Source: 自建fq节点.mp4 (19-min Chinese tutorial, natural spoken English terms:
VPS/SSH/TLS/Cloudflare/Shadowsocks...). Cuts 20s segments from the word-level SRT,
runs both engines, reports han-CER and mixed-CER — the spoken counterpart of the
FLEURS read-speech mixed subset.

Usage: python runners/bench_spoken_mixed.py --engines glm,fun [--max-seg 30]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.models import get_engine
from runners.bench_mixed import han_units, mixed_units, edit_distance_units

ROOT = Path(__file__).resolve().parent.parent
VIDEO = Path("/data/其他模型/小型语言模型/自建fq节点.mp4")
SRT = Path("/data/其他模型/小型语言模型/自建fq节点.srt")


def parse_srt(path: Path):
    entries = []
    ts = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")
    text, start, end = [], None, None
    for line in path.read_text(encoding="utf-8").splitlines() + [""]:
        m = ts.search(line)
        if m:
            g = [int(x) for x in m.groups()]
            start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
            end = g[3 + 0] * 0 + g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
            text = []
        elif line.strip() and start is not None and not line.strip().isdigit():
            text.append(line.strip())
        elif not line.strip() and start is not None:
            entries.append((start, end, " ".join(text)))
            start = None
    return entries


def cut_wav(seg_start: float, seg_end: float, tmp: Path):
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{seg_start:.2f}", "-to", f"{seg_end:.2f}",
         "-i", str(VIDEO), "-ar", "16000", "-ac", "1", str(tmp)],
        check=True, capture_output=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="glm,fun")
    ap.add_argument("--gpu", action="store_true", default=True)
    ap.add_argument("--cpu", dest="gpu", action="store_false")
    ap.add_argument("--seg-len", type=float, default=20.0)
    ap.add_argument("--max-seg", type=int, default=30)
    ap.add_argument("--out", default="results/spoken_mixed.json")
    args = ap.parse_args()

    import soundfile as sf

    entries = parse_srt(SRT)
    dur = 1161.14

    segments = []
    t = 0.0
    while t + args.seg_len <= dur and len(segments) < args.max_seg:
        seg_end = t + args.seg_len
        refs = [txt for (s, e, txt) in entries if s >= t - 0.5 and e <= seg_end + 0.5]
        segments.append((t, seg_end, " ".join(refs)))
        t += args.seg_len

    EN = re.compile(r"[a-zA-Z]{2,}")
    HAN = re.compile(r"[\u4e00-\u9fff]")
    mixed_segs = [(s, e, r) for s, e, r in segments if HAN.search(r) and EN.search(r)]
    print(f"segments: {len(segments)} total, {len(mixed_segs)} mixed ({len(mixed_segs)/max(len(segments),1)*100:.0f}%)")

    engines = {}
    for name in [e.strip() for e in args.engines.split(",")]:
        kwargs = {"use_gpu": args.gpu} if name == "glm" else {}
        engines[name] = get_engine(name, **kwargs)

    stats = {}
    with tempfile.TemporaryDirectory() as td:
        for i, (s, e, ref) in enumerate(mixed_segs):
            wav = Path(td) / f"seg_{i}.wav"
            cut_wav(s, e, wav)
            audio, _ = sf.read(wav, dtype="float32")
            for name, engine in engines.items():
                hyp = engine.decode_text(engine.encode(audio))
                he, ht = edit_distance_units(han_units(ref.lower()), han_units(hyp))
                me, mt = edit_distance_units(mixed_units(ref.lower()), mixed_units(hyp))
                st = stats.setdefault(name, [0, 0, 0, 0])
                st[0] += he; st[1] += ht; st[2] += me; st[3] += mt
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(mixed_segs)} segs")

    import json
    out = {}
    for name, (he, ht, me, mt) in stats.items():
        out[name] = {
            "cer_han": round(he / max(ht, 1) * 100, 1),
            "mixed_cer": round(me / max(mt, 1) * 100, 1),
            "han_err/total": [he, ht],
            "mixed_err/total": [me, mt],
        }
        print(f"[{name}] han-CER={out[name]['cer_han']}%  mixed-CER={out[name]['mixed_cer']}%  ({he}/{ht}, {me}/{mt})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"n_segments": len(mixed_segs), **out}, ensure_ascii=False, indent=2))
    print(f"results -> {args.out}")


if __name__ == "__main__":
    main()
