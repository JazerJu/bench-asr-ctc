#!/usr/bin/env python3
"""Prepare untracked audio for the two public long-video case studies."""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def extract(source: Path, output: Path, start: float = 0, duration: float | None = None):
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-n", "-ss", str(start), "-i", str(source)]
    if duration is not None:
        cmd += ["-t", str(duration)]
    subprocess.run(cmd + ["-vn", "-ac", "1", "-ar", "16000",
                          "-c:a", "pcm_s16le", str(output)], check=True)


def prepare(case: Path, source: Path):
    if not source.is_file():
        raise FileNotFoundError(f"source media not found: {source}")
    spec = json.loads((case / "source.json").read_text(encoding="utf-8"))
    for filename, cut in spec["outputs"].items():
        output = case / filename
        if output.exists():
            print(f"already present: {output}")
            continue
        extract(source, output, cut.get("start", 0), cut.get("duration"))
        print(f"prepared: {output}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("case", choices=["ascend_cube", "tedx_fukuoka"])
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--source", type=Path, help="local copy of the original video/audio")
    group.add_argument("--download", action="store_true",
                       help="explicitly download the public source using yt-dlp")
    args = ap.parse_args()
    case = ROOT / "cases" / args.case
    if args.source:
        prepare(case, args.source.expanduser().resolve())
        return
    spec = json.loads((case / "source.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="ctc-case-") as td:
        media = Path(td) / "source.%(ext)s"
        subprocess.run(["yt-dlp", "--no-playlist", "-f", "bestaudio/best",
                        "-o", str(media), spec["url"]], check=True)
        sources = [p for p in Path(td).glob("source.*") if p.suffix != ".part"]
        if len(sources) != 1:
            raise RuntimeError(f"expected one source file, got {sources}")
        prepare(case, sources[0])


if __name__ == "__main__":
    main()
