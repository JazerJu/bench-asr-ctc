#!/usr/bin/env python3
"""Single-audio inference: GLM-ASR-CTC or Fun-ASR-Nano backend, any input ffmpeg reads.

CTC first-pass only (no LLM decoder, no hotwords) — for full-pipeline transcription
with LLM correction use the GLM-ASR-CTC-GGUF project's main.py.

Examples:
    python infer.py clip.wav
    python infer.py input.mp3 --engine fun
    python infer.py recording.m4a --engine glm --cpu
"""

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench.models import get_engine


def to_wav16k(src: Path, tmpdir: Path) -> Path:
    dst = tmpdir / "input16k.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ar", "16000", "-ac", "1", str(dst)],
        check=True, capture_output=True,
    )
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", help="input audio (any ffmpeg-readable format, designed for ≤30s)")
    ap.add_argument("--engine", default="glm", choices=["glm", "fun", "qwen"], help="CTC backend")
    ap.add_argument("--cpu", action="store_true", help="run ONNX on CPU (default CUDA when available)")
    args = ap.parse_args()

    src = Path(args.audio).expanduser()
    if not src.exists():
        raise SystemExit(f"input not found: {src}")

    engine = get_engine(args.engine, use_gpu=not args.cpu)

    with tempfile.TemporaryDirectory() as td:
        wav = to_wav16k(src, Path(td))
        import soundfile as sf

        audio, sr = sf.read(wav, dtype="float32")
        duration = len(audio) / sr
        if duration > 30.5:
            print(f"[warn] {duration:.1f}s exceeds the 30s design point; running anyway", file=sys.stderr)

        t0 = time.perf_counter()
        text = engine.decode_text(engine.encode(audio))
        wall = time.perf_counter() - t0

    print(f"[{args.engine}] {text}")
    print(f"[stats] {duration:.1f}s audio -> {wall*1000:.0f}ms (RTF {wall/duration:.3f}, "
          f"{'CPU' if args.cpu else engine.enc_sess.get_providers()[0]})")


if __name__ == "__main__":
    main()
