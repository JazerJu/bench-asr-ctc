#!/usr/bin/env python3
"""B 站 TED 合集（BV1pBp46rEQh，29P）：各 CTC 头在纯英文音频上串出中文的情况，以及 TED-LIUM 污染核对。

issue #457 里 H1DDENADM1N 用 aiinfra4 跑了整个合集，23/29P 的 SRT 里出现中文。
他的 SRT 文本来自 CTC 首遍，所以这里统计 CTC 首遍里的汉字；同时确认 LLM 二遍最终文本没有汉字。

污染：aiinfra5 训练里加了 TED-LIUM 3（只剔除了 ReshmaSaujani_2016）。这个合集是 TED 演讲，
可能就在训练集里。从每一 P 的最终转写随机抽 8 个"连续 8 词"窗口，到 TED-LIUM 训练文本里精确检索。

    python ted_collection_check.py --head aiinfra5=<onnx>          # 每个头单独一个进程（两个引擎同进程会崩）
    python ted_collection_check.py --report v1,aiinfra4,aiinfra5 --tedlium /tmp/tedlium3_texts.tsv
"""
import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASE = ROOT / "cases" / "ted_collection"
VIDEO_DIR = Path(os.environ.get("VIDGO_VIDEO_DIR", "/media/jju/ExtraDisk/VidGo/backend/media/saved_video"))
HAN = re.compile(r"[㐀-鿿豈-﫿]+")


def list_parts(manifest: Path = CASE / "clips.tsv", vidgo_api: str | None = None):
    if vidgo_api is None:
        with manifest.open(encoding="utf-8") as stream:
            return {int(r["part"]): (r["vidgo_hash"], r["title"])
                    for r in csv.DictReader(stream, delimiter="\t")}
    import urllib.request
    d = json.load(urllib.request.urlopen(vidgo_api.rstrip("/") + "/api/videos/", timeout=60))
    out = {}

    def walk(o):
        if isinstance(o, dict):
            u = str(o.get("url", ""))
            m = re.search(r"精选30篇经典TED演讲全解析.*?-p(\d+)-", o.get("name", "") or "")
            if u.endswith(".mp4") and m:
                out[int(m.group(1))] = (u[:-4], o["name"].split(f"-p{m.group(1)}-", 1)[1])
                return
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(d)
    return dict(sorted(out.items()))


def ensure_wav(p, vhash, video_dir: Path = VIDEO_DIR):
    wav = CASE / "audio" / f"p{p:02d}.wav"
    if not wav.exists():
        source = video_dir / f"{vhash}.mp4"
        if not source.is_file():
            raise FileNotFoundError(f"Missing {wav}; obtain the original media and use --video-dir, "
                                    f"or prepare this mono 16 kHz WAV. See {CASE / 'README.md'}.")
        wav.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(source), "-vn", "-ac", "1",
                        "-ar", "16000", "-c:a", "pcm_s16le", str(wav)], check=True)
    return wav


def norm_words(s):
    return re.sub(r"[^a-z0-9' ]+", " ", s.lower().replace("’", "'")).split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", help="label=onnx 路径：跑这一个头，结果存 results/ted_collection_<label>.json")
    ap.add_argument("--report", help="逗号分隔的 label：汇总已跑完的头并做污染核对")
    ap.add_argument("--tedlium", type=Path)
    ap.add_argument("--capswriter", type=Path, default=Path("/data/AI应用/流式转录/CapsWriter-Offline"))
    ap.add_argument("--manifest", type=Path, default=CASE / "clips.tsv")
    ap.add_argument("--vidgo-api", help="optional: discover parts from this live VidGo service instead of the manifest")
    ap.add_argument("--video-dir", type=Path, default=VIDEO_DIR)
    a = ap.parse_args()
    if not a.head and not a.report:
        ap.error("provide --head or --report")
    if a.report and not a.head and not a.tedlium:
        ap.error("--report requires --tedlium")

    parts = list_parts(a.manifest, a.vidgo_api)
    if not parts:
        raise ValueError("no TED parts found")
    print(f"合集里找到 {len(parts)} P：{sorted(parts)}", flush=True)

    if a.head:
        wavs = {p: ensure_wav(p, h, a.video_dir) for p, (h, _) in parts.items()}
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from en_regression import make_engine

        label, head = a.head.split("=", 1)
        eng = make_engine(a.capswriter, Path(head), with_decoder=True)
        one = {}
        for p, wav in wavs.items():
            r = eng.transcribe(str(wav))
            spans = HAN.findall(r.ctc_text)
            one[p] = {"ctc_han": sum(len(s) for s in spans), "spans": spans,
                      "final_han": len(HAN.findall(r.text)), "final_text": r.text}
        json.dump({"head": head, "parts": one}, open(ROOT / "results" / f"ted_collection_{label}.json", "w",
                  encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"{label}: CTC 首遍汉字共 {sum(v['ctc_han'] for v in one.values())} 个，"
              f"{sum(v['ctc_han'] > 0 for v in one.values())}/{len(wavs)} P 出现中文", flush=True)
        return

    res = {}
    for label in a.report.split(","):
        d = json.load(open(ROOT / "results" / f"ted_collection_{label}.json", encoding="utf-8"))["parts"]
        res[label] = {int(k): v for k, v in d.items()}

    # 污染核对：用第一个头的最终文本（各头的最终文本相同）
    train_ngr, full_ngr = set(), set()
    for line in open(a.tedlium, encoding="utf-8"):
        tag, spk, text = line.rstrip("\n").split("\t", 2)
        w = norm_words(text)
        target = train_ngr if tag == "train" else full_ngr
        for i in range(len(w) - 7):
            target.add(" ".join(w[i:i + 8]))
    rnd = random.Random(0)
    first = next(iter(res.values()))
    contam = {}
    for p in parts:
        w = norm_words(first[p]["final_text"])
        wins = [" ".join(w[i:i + 8]) for i in rnd.sample(range(max(1, len(w) - 7)), min(8, max(1, len(w) - 7)))]
        contam[p] = (sum(x in train_ngr for x in wins), sum(x in full_ngr for x in wins), len(wins))

    labels = list(res)
    print("\n%-4s %-34s %s  %s" % ("P", "标题", "  ".join("%9s" % l for l in labels), "TED-LIUM 命中（a5 训练用 / 完整版）"))
    for p, (h, title) in parts.items():
        c = contam[p]
        print("%-4s %-34s %s  %d/%d  %d/%d" % (f"P{p}", title[:17], "  ".join("%9d" % res[l][p]["ctc_han"] for l in labels),
                                            c[0], c[2], c[1], c[2]))
    print("\n最终文本（LLM 二遍）汉字：" + "  ".join(f"{l} {sum(v['final_han'] for v in res[l].values())}" for l in labels))

    clean = [p for p in parts if contam[p][0] == 0]
    print(f"\n抽样窗口未命中训练文本的 P（不能据此证明未进训练）：{['P%d' % p for p in clean]}")
    for l in labels:
        print(f"  {l}: 这些 P 上 CTC 汉字共 {sum(res[l][p]['ctc_han'] for p in clean)}，"
              f"{sum(res[l][p]['ctc_han'] > 0 for p in clean)}/{len(clean)} P 出现中文")

    out = ROOT / "results" / "ted_collection_check.json"
    json.dump({"heads": a.report, "contamination": {p: contam[p] for p in contam},
               "results": {l: {p: {k: v for k, v in d.items() if k != "final_text"} for p, d in res[l].items()} for l in res}},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("结果 ->", out)


if __name__ == "__main__":
    main()
