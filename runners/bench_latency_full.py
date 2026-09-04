#!/usr/bin/env python3
"""三家全流程延迟：Fun / Qwen3-CTC / GLM，同一进程、同一段音频、各自的正式推理包。

每家都跑完整 pipeline（编码 -> CTC 首遍 -> 热词 -> LLM -> 时间戳对齐），从各引擎自带
的 Timings 里拆出分段：
    CTC 列        = encode + ctc          （首遍能出字的时刻）
    加 Decoder 列 = 整条 decode_stream    （含 prefill / 生成 / 对齐）
全 GPU：ONNX CUDA EP + llama.cpp CUDA。**llama 必须先于 ORT CUDA 初始化**，反过来
同进程 SIGSEGV —— 三个包的 engine 都已经按这个顺序写好，这里只要按 Fun -> Qwen -> GLM
依次构造即可（每家构造时都是先 llama 后 ORT）。

    for e in fun qwen glm; do python runners/bench_latency_full.py --engines $e; done
每次只跑 --engines 指定的引擎，结果合并进 results/latency_3way.json（同进程连跑三家会在
第三家 abort，见 main 里的注释）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import torch  # noqa: F401  — 训练 venv 里 onnxruntime 的 CUDA EP 依赖 torch 预加载的 libcufft.so.12，
except Exception:  #             不先 import torch 会静默掉回 CPU EP
    pass
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
CW_ROOT = Path("/data/AI应用/流式转录/CapsWriter-Offline")
FUN_MODELS = CW_ROOT / "models/Fun-ASR-Nano/Fun-ASR-Nano-GGUF"
GLM_ROOT = Path("/data/推理框架/asr-onnx/GLM-ASR-CTC-GGUF")
QWEN_ROOT = Path("/data/推理框架/asr-onnx/Qwen3-ASR-CTC-GGUF")
FUN_MODELS = Path("/data/AI应用/流式转录/CapsWriter-Offline/models/Fun-ASR-Nano/Fun-ASR-Nano-GGUF")
N_WARM, N_RUN = 1, 5
SECS = (5, 30)


def load_audio(path: Path):
    wav, sr = sf.read(str(path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    assert sr == 16000, sr
    return wav


def clip(wav, seconds):
    need = int(seconds * 16000)
    if len(wav) >= need:
        return wav[:need]
    return np.tile(wav, (need + len(wav) - 1) // len(wav))[:need]


def measure(engine, audio, temperature=0.0):
    """返回 (中位总耗时 ms, 最后一次的 DecodeResult)。"""
    def run():
        st = engine.create_stream()
        st.accept_waveform(16000, audio)
        return engine.decode_stream(st, verbose=False, temperature=temperature)
    for _ in range(N_WARM):
        run()
    ts, res = [], None
    for _ in range(N_RUN):
        t0 = time.perf_counter()
        res = run()
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return ts[len(ts) // 2], res


def breakdown(res):
    T = res.timings
    ms = lambda x: round(x * 1000, 1)
    return {
        "encode": ms(T.encode), "ctc": ms(T.ctc), "inject": ms(T.inject),
        "generate": ms(T.llm_generate), "align": ms(T.align),
        "ctc_only": ms(T.encode + T.ctc),
        "n_gen": int(getattr(res, "n_gen", getattr(res, "n_generated_tokens", 0))),
        "text": res.text[:40],
    }


def build_fun():
    """CapsWriter 部署的 Fun：util/fun_asr_gguf 引擎 + models/ 里发布的 int4 ONNX + 它自带的 llama.cpp 库。
    Fun-ASR-GGUF 仓库里的代码要 CTC ONNX 出两个输出，和 CapsWriter 里只出 indices 的模型对不上，所以走 CW 的引擎。"""
    sys.path.insert(0, str(CW_ROOT))
    from util.fun_asr_gguf.inference.asr_engine import create_asr_engine
    return create_asr_engine(
        encoder_onnx_path=str(FUN_MODELS / "Fun-ASR-Nano-Encoder-Adaptor.int4.onnx"),
        ctc_onnx_path=str(FUN_MODELS / "Fun-ASR-Nano-CTC.int4.onnx"),
        decoder_gguf_path=str(FUN_MODELS / "Fun-ASR-Nano-Decoder.q5_k.gguf"),
        tokens_path=str(FUN_MODELS / "tokens.txt"),
        hotwords_path=None, enable_ctc=True, n_predict=256, dml_enable=False, verbose=False)


def build_qwen():
    sys.path.insert(0, str(QWEN_ROOT))
    from qwen3_asr_ctc import create_asr_engine
    return create_asr_engine(
        encoder_onnx_path=str(QWEN_ROOT / "model/Qwen3-ASR-Encoder.q4.onnx"),   # 表按发布的 q4 计；CUDA 上 fp16 / q4f16 快一倍，见 README
        ctc_onnx_path=str(QWEN_ROOT / "model/Qwen3-ASR-CTC.q4.onnx"),
        tokens_path=str(QWEN_ROOT / "model/tokens.txt"),
        preprocessor_path=str(QWEN_ROOT / "preprocessor"),
        decoder_gguf_path=str(QWEN_ROOT / "model/Qwen3-ASR-Decoder.q5_k_m.gguf"),
        n_predict=256)


def build_glm():
    sys.path.insert(0, str(GLM_ROOT))
    from glm_asr_ctc.engine import ASREngineConfig, GLMASREngine
    snaps = sorted(Path("/data/.cache/huggingface/hub/models--zai-org--GLM-ASR-Nano-2512/snapshots").glob("*"))
    return GLMASREngine(ASREngineConfig(
        model_id=str(snaps[-1]) if snaps else "",
        encoder_onnx_path=str(GLM_ROOT / "model/GLM-ASR-Encoder.q4.onnx"),
        projector_onnx_path=str(GLM_ROOT / "model/GLM-ASR-Projector.fp16.onnx"),
        ctc_onnx_path=str(GLM_ROOT / "model/GLM-ASR-CTC-Final134k.q4.onnx"),
        decoder_gguf_path=str(GLM_ROOT / "model/GLM-ASR-Nano-Decoder.q5_k_m.gguf"),
        tokens_path=str(GLM_ROOT / "model/tokens-phase2.txt"),
        onnx_provider="CUDA", llm_use_gpu=True, verbose=False, n_predict=256))


BUILDERS = {"fun": build_fun, "qwen": build_qwen, "glm": build_glm}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wav", default=str(ROOT / "cases/zh_news_30s.wav"))
    ap.add_argument("--engines", default="fun,qwen,glm")
    ap.add_argument("--out", default=str(ROOT / "results/latency_3way.json"))
    args = ap.parse_args()
    wav = load_audio(Path(args.wav))
    # 三家各起一个进程再合并：同进程里连跑三个 llama.cpp 上下文，第三个（GLM 30s）会在
    # ggml-cuda.cu:103 abort；分进程还能保证每家测的时候 GPU 状态干净。
    prev = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}
    result = {
        "gpu": "RTX 5070 Ti, warm median of 5, published int4/q4 ONNX (CUDA EP, fp32 activations, ArgMax in CTC graph) + q5 decoder (llama.cpp CUDA)",
        "engines": {"fun": "CapsWriter util/fun_asr_gguf + models/Fun-ASR-Nano-GGUF int4 (bundled llama.cpp)", "qwen": "Qwen3-ASR-CTC-GGUF/qwen3_asr_ctc", "glm": "GLM-ASR-CTC-GGUF/glm_asr_ctc"},
        "wav": str(Path(args.wav).relative_to(ROOT)) if Path(args.wav).is_relative_to(ROOT) else args.wav,
        "definition": {"ctc_only_ms": "encode + ctc", "full_pipeline_ms": "whole decode_stream incl. align"},
        "ctc_only_ms": {}, "full_pipeline_ms": {}, "breakdown_ms": {},
    }
    for k in ("ctc_only_ms", "full_pipeline_ms", "breakdown_ms"):
        result[k].update(prev.get(k, {}))
    for name in args.engines.split(","):
        print(f"== {name} ==", flush=True)
        eng = BUILDERS[name]()
        result["ctc_only_ms"][name] = {}
        result["full_pipeline_ms"][name] = {}
        result["breakdown_ms"][name] = {}
        for sec in SECS:
            total, res = measure(eng, clip(wav, sec))
            b = breakdown(res)
            result["ctc_only_ms"][name][f"{sec}s"] = round(b["ctc_only"])
            result["full_pipeline_ms"][name][f"{sec}s"] = round(total)
            result["breakdown_ms"][name][f"{sec}s"] = b
            print(f"  {sec:>2}s  total={total:7.1f}  enc={b['encode']:6.1f} ctc={b['ctc']:6.1f} "
                  f"inject={b['inject']:5.1f} gen={b['generate']:6.1f} align={b['align']:6.1f} "
                  f"n_gen={b['n_gen']}  {b['text']!r}", flush=True)
        try:
            eng.cleanup()
        except Exception:
            pass
    Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print("wrote", args.out)
    print("\n|          | CTC 5s | CTC 30s | 加 Decoder 5s | 加 Decoder 30s |")
    print("| -------- | ------ | ------- | ------------- | -------------- |")
    label = {"fun": "Fun", "qwen": "Qwen-CTC", "glm": "GLM"}
    for name in [n for n in ("fun", "qwen", "glm") if n in result["full_pipeline_ms"]]:
        c, f = result["ctc_only_ms"][name], result["full_pipeline_ms"][name]
        print(f"| {label[name]:<8} | {c['5s']:>6} | {c['30s']:>7} | {f['5s']:>13} | {f['30s']:>14} |")


if __name__ == "__main__":
    main()
