#!/usr/bin/env python3
"""英文回归集：一个 CTC 头在 6 条英文音频上的丢词、CER 和 SRT 时间戳偏移。

aiinfra4 在部分英文音频上 CTC 首遍丢词，NW 对齐缺锚点，SRT 时间戳成段偏移。
FLEURS / LibriSpeech 都是朗读短句，抓不到这个问题，所以单独固定这一组。

参考 = v1 头 + LLM 二遍的完整输出（cases/en_regression/refs/*.json，含最终文本和逐词时间戳）。
LLM 二遍的最终文本与 CTC 头无关（CTC 文本只用来抽热词，不进 prompt），所以各个头的
最终文本逐字相同，切出来的字幕条数也相同，可以按序号逐条比时间。

指标：
  CTC 字数比   CTC 首遍字符数 / 参考字符数（只数 a-z0-9），丢词时明显下降
  CER          CTC 首遍对参考的字符错误率
  汉字         CTC 首遍里的汉字个数
  SRT >1s/>2s  完整流水线的字幕条，起始时间与 v1 基线相差超过 1 / 2 秒的条数

用法（在 .92 上，引擎用 CapsWriter-Offline 里的 util/qwen3_ctc_gguf）：
    python scripts/en_regression.py --head <Qwen3-ASR-CTC.*.onnx> --label aiinfra5
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CASE = ROOT / "cases" / "en_regression"
VIDEO_DIR = Path("/media/jju/ExtraDisk/VidGo/backend/media/saved_video")


def to_chars(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def edit_distance(a, b) -> int:
    n, m = len(a), len(b)
    prev = np.arange(m + 1, dtype=np.int32)
    k = np.arange(m + 1, dtype=np.int32)
    b = np.array(b, dtype=object)
    for i in range(1, n + 1):
        cost = (b != a[i - 1]).astype(np.int32)
        row = np.empty(m + 1, dtype=np.int32)
        row[0] = i
        row[1:] = np.minimum(prev[1:] + 1, prev[:-1] + cost)
        prev = k + np.minimum.accumulate(row - k)
    return int(prev[m])


def cues(words, max_sec=12.0, max_chars=90):
    """按句末标点 / 12 秒 / 90 字符切字幕条，和 opencode 那轮实测的 srt_gen.py 一致。"""
    out, cur = [], []
    for w, s, e in words:
        if not w.strip():
            if cur:
                cur.append((" ", s, e))
            continue
        cur.append((w, s, e))
        text = "".join(x[0] for x in cur)
        if w.strip() in ".!?" or (cur[-1][2] - cur[0][1] >= max_sec) or len(text) >= max_chars:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def ensure_wav(key: str, vhash: str) -> Path:
    wav = CASE / "audio" / f"{key}.wav"
    if not wav.exists():
        wav.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(VIDEO_DIR / f"{vhash}.mp4"),
                        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)], check=True)
    return wav


def make_engine(capswriter: Path, head: Path, with_decoder: bool):
    sys.path.insert(0, str(capswriter))
    from util.qwen3_ctc_gguf import create_asr_engine
    m = capswriter / "models" / "Qwen3-ASR-CTC"
    return create_asr_engine(
        encoder_onnx_path=str(m / "Qwen3-ASR-Encoder.q4f16.onnx"),
        ctc_onnx_path=str(head),
        tokens_path=str(m / "qwen-ctc-tokens.txt"),
        preprocessor_path=str(m),
        use_gpu=True,
        llm_use_gpu=True,
        decoder_gguf_path=str(m / "Qwen3-ASR-Decoder.q5_k_m.gguf") if with_decoder else None,
        hotwords=[],
        ort_gpu_mem_limit_mb=1800,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True, type=Path, help="CTC 头 ONNX")
    ap.add_argument("--label", required=True)
    ap.add_argument("--capswriter", type=Path,
                    default=Path(os.environ.get("CAPSWRITER_ROOT", "/data/AI应用/流式转录/CapsWriter-Offline")))
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    clips = list(csv.DictReader(open(CASE / "clips.tsv", encoding="utf-8"), delimiter="\t"))
    wavs = {c["key"]: ensure_wav(c["key"], c["vidgo_hash"]) for c in clips}
    refs = {c["key"]: json.load(open(CASE / "refs" / f"{c['key']}.json", encoding="utf-8")) for c in clips}
    han = re.compile(r"[㐀-鿿豈-﫿]")

    # 只建一个带 decoder 的引擎：结果里同时有 CTC 首遍文本和逐词时间戳。
    # 先建纯 CTC 引擎再建带 decoder 的会崩（llama.cpp 必须先于 ORT CUDA 初始化）。
    eng = make_engine(a.capswriter, a.head, with_decoder=True)
    rows = []
    for c in clips:
        k = c["key"]
        r = eng.transcribe(str(wavs[k]))
        hyp = r.ctc_text
        rc, hc = to_chars(refs[k]["final_text"]), to_chars(hyp)
        row = {"clip": k, "ctc_ratio": len(hc) / max(1, len(rc)),
               "cer": edit_distance(list(rc), list(hc)) / max(1, len(rc)),
               "han": len(han.findall(hyp)), "ctc_text": hyp}
        rows.append(row)
        ref_c, hyp_c = cues(refs[k]["words"]), cues(r.words)
        row["final_same"] = r.text == refs[k]["final_text"]
        row["cues"] = len(ref_c)
        if len(ref_c) != len(hyp_c):
            row["srt_gt1"] = row["srt_gt2"] = None
            row["note"] = f"字幕条数不同（{len(hyp_c)} vs {len(ref_c)}），没法逐条比"
            continue
        drift = [abs(h[0][1] - g[0][1]) for h, g in zip(hyp_c, ref_c)]
        row["srt_gt1"] = sum(d > 1.0 for d in drift)
        row["srt_gt2"] = sum(d > 2.0 for d in drift)
        row["srt_max"] = max(drift) if drift else 0.0

    print(f"\n== {a.label}  {a.head}")
    print(f"{'clip':<7s}{'CTC字数比':>10s}{'CER':>8s}{'汉字':>6s}{'SRT>1s':>12s}{'SRT>2s':>8s}{'最大偏移':>10s}  最终文本同 v1")
    for r in rows:
        g1 = "-" if r["srt_gt1"] is None else f"{r['srt_gt1']}/{r['cues']}"
        g2 = "-" if r["srt_gt2"] is None else str(r["srt_gt2"])
        mx = "-" if r.get("srt_max") is None else f"{r['srt_max']:.1f}s"
        print(f"{r['clip']:<7s}{r['ctc_ratio'] * 100:9.1f}%{r['cer'] * 100:7.1f}%{r['han']:6d}{g1:>12s}{g2:>8s}{mx:>10s}  {r['final_same']}"
              + (f"  {r['note']}" if r.get("note") else ""))

    out = a.out or ROOT / "results" / f"en_regression_{a.label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"label": a.label, "head": str(a.head), "rows": rows}, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("结果 ->", out)


if __name__ == "__main__":
    main()
