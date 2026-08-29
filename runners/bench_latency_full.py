#!/usr/bin/env python3
"""Fair full-pipeline latency: Fun / Qwen / GLM, all CUDA ONNX + CUDA llama."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
FUN_ROOT = Path("/data/推理框架/asr-onnx/Fun-ASR-GGUF")
GLM_ROOT = Path("/data/推理框架/asr-onnx/GLM-ASR-CTC-GGUF")
QWEN_ROOT = Path("/data/推理框架/asr-onnx/Qwen3-ASR-CTC-GGUF")
FUN_MODELS = Path("/data/AI应用/流式转录/CapsWriter-Offline/models/Fun-ASR-Nano/Fun-ASR-Nano-GGUF")
WAV = Path("/tmp/cmdcpu_test.wav")
N_WARM, N_RUN = 1, 5


def _audio(seconds: float) -> np.ndarray:
    wav, sr = sf.read(str(WAV), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    need = int(seconds * sr)
    if len(wav) >= need:
        return wav[:need]
    reps = (need + len(wav) - 1) // len(wav)
    return np.tile(wav, reps)[:need]


def _median_ms(fn, n_warm=N_WARM, n_run=N_RUN) -> float:
    for _ in range(n_warm):
        fn()
    ts = []
    for _ in range(n_run):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


def bench_fun():
    sys.path.insert(0, str(FUN_ROOT))
    from fun_asr_gguf import ASREngineConfig, FunASREngine

    cfg = ASREngineConfig(
        encoder_onnx_path=str(FUN_MODELS / "Fun-ASR-Nano-Encoder-Adaptor.int4.onnx"),
        ctc_onnx_path=str(FUN_MODELS / "Fun-ASR-Nano-CTC.int4.onnx"),
        decoder_gguf_path=str(FUN_MODELS / "Fun-ASR-Nano-Decoder.q5_k.gguf"),
        tokens_path=str(FUN_MODELS / "tokens.txt"),
        onnx_provider="CUDA",
        llm_use_gpu=True,
        enable_ctc=True,
        verbose=False,
        n_predict=256,
    )
    engine = FunASREngine(cfg)
    out = {}
    for sec in (5, 30):
        audio = _audio(sec)

        def run(a=audio):
            stream = engine.create_stream()
            stream.accept_waveform(16000, a)
            return engine.decode_stream(stream, verbose=False, temperature=0.0)

        text = run().text
        out[sec] = round(_median_ms(run))
        print(f"  fun {sec}s  {out[sec]} ms  text[:60]={text[:60]!r}")
    engine.cleanup()
    return out


def bench_qwen():
    sys.path.insert(0, str(GLM_ROOT))
    import onnxruntime as ort
    from transformers import WhisperFeatureExtractor
    from glm_asr_ctc import llama_cpp_bindings as llama

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    enc = ort.InferenceSession(str(QWEN_ROOT / "model/Qwen3-ASR-Encoder.q4.onnx"), providers=providers)
    ctc = ort.InferenceSession(str(QWEN_ROOT / "model/Qwen3-ASR-CTC.q4.onnx"), providers=providers)
    fe = WhisperFeatureExtractor.from_pretrained(str(QWEN_ROOT / "preprocessor"), local_files_only=True)
    model = llama.LlamaModel(str(QWEN_ROOT / "model/Qwen3-ASR-Decoder.q5_k_m.gguf"), use_gpu=True)
    ctx = llama.LlamaContext(model, n_ctx=4096, n_batch=4096, n_ubatch=512)
    table = llama.get_token_embeddings_gguf(str(QWEN_ROOT / "model/Qwen3-ASR-Decoder.q5_k_m.gguf"))

    def tok_id(s):
        ids = model.tokenize(s, add_special=False, parse_special=True)
        return ids[0] if ids else -1

    ID_IM_START = tok_id("<|im_start|>")
    ID_IM_END = tok_id("<|im_end|>")
    ID_AUDIO_START = tok_id("<|audio_start|>")
    ID_AUDIO_END = tok_id("<|audio_end|>")
    ID_ASR_TEXT = tok_id("<asr_text>")
    print(f"  qwen specials start={ID_IM_START} audio={ID_AUDIO_START} asr={ID_ASR_TEXT} n_embd={model.n_embd}")

    def encode(audio):
        mel = fe(audio, sampling_rate=16000, padding=False, return_tensors="np").input_features[0]
        T = mel.shape[1]
        padded = np.zeros((128, 3000), dtype=np.float32)
        padded[:, :T] = mel
        hidden = enc.run(None, {"input_features": padded, "feature_length": np.array([T], np.int64)})[0]
        full, leave = divmod(T, 100)
        valid = full * 13 + (0 if leave == 0 else (leave - 1) // 8 + 1)
        audio_embd = hidden[:valid].astype(np.float32)
        _ = ctc.run(None, {"enc_output": audio_embd[None]})
        return audio_embd

    def transcribe(audio):
        audio_embd = encode(audio)
        prefix = [ID_IM_START] + model.tokenize("system\nYou are a helpful assistant.") + [ID_IM_END]
        prefix += [ID_IM_START] + model.tokenize("user\n") + [ID_AUDIO_START]
        suffix = [ID_AUDIO_END, ID_IM_END, ID_IM_START] + model.tokenize("assistant\n") + [ID_ASR_TEXT]
        n_pre, n_aud, n_suf = len(prefix), audio_embd.shape[0], len(suffix)
        full = np.zeros((n_pre + n_aud + n_suf, model.n_embd), np.float32)
        full[:n_pre] = table[prefix]
        full[n_pre:n_pre + n_aud] = audio_embd
        full[n_pre + n_aud:] = table[suffix]
        n = full.shape[0]
        pos = np.arange(n, dtype=np.int32)
        pos_arr = np.concatenate([pos, pos, pos, np.zeros(n, dtype=np.int32)])
        ctx.clear_kv_cache()
        batch = llama.LlamaBatch(max(n * 4, 8192), model.n_embd, 1)
        batch.set_embd(full, pos=pos_arr)
        if ctx.decode(batch) != 0:
            raise RuntimeError("qwen prefill failed")
        pieces = []
        with llama.LlamaSampler(temperature=0.0, top_k=1, top_p=1.0, seed=0) as smpl:
            for _ in range(256):
                tid = smpl.sample(ctx, -1)
                if tid in (model.eos_token, ID_IM_END):
                    break
                if ctx.decode_token(tid) != 0:
                    break
                pieces.append(tid)
        return model.detokenize(pieces)

    out = {}
    for sec in (5, 30):
        audio = _audio(sec)
        text = transcribe(audio)
        out[sec] = round(_median_ms(lambda a=audio: transcribe(a)))
        print(f"  qwen {sec}s  {out[sec]} ms  text[:60]={text[:60]!r}")
    return out


def bench_glm():
    sys.path.insert(0, str(GLM_ROOT))
    from glm_asr_ctc.engine import ASREngineConfig, GLMASREngine

    snaps = sorted(Path("/data/.cache/huggingface/hub/models--zai-org--GLM-ASR-Nano-2512/snapshots").glob("*"))
    cfg = ASREngineConfig(
        model_id=str(snaps[-1]) if snaps else "",
        encoder_onnx_path=str(GLM_ROOT / "model/GLM-ASR-Encoder.q4.onnx"),
        projector_onnx_path=str(GLM_ROOT / "model/GLM-ASR-Projector.fp16.onnx"),
        ctc_onnx_path=str(GLM_ROOT / "model/GLM-ASR-CTC-Final134k.q4.onnx"),
        decoder_gguf_path=str(GLM_ROOT / "model/GLM-ASR-Nano-Decoder.q5_k_m.gguf"),
        tokens_path=str(GLM_ROOT / "model/tokens-phase2.txt"),
        onnx_provider="CUDA",
        llm_use_gpu=True,
        verbose=False,
        n_predict=256,
    )
    engine = GLMASREngine(cfg)
    out = {}
    for sec in (5, 30):
        audio = _audio(sec)

        def run(a=audio):
            stream = engine.create_stream()
            stream.accept_waveform(16000, a)
            return engine.decode_stream(stream, verbose=False, temperature=0.0)

        text = run().text
        out[sec] = round(_median_ms(run))
        print(f"  glm {sec}s  {out[sec]} ms  text[:60]={text[:60]!r}")
    return out


def main():
    print("== Fun CUDA ONNX + CUDA llama ==")
    fun = bench_fun()
    print("== Qwen CUDA ONNX + CUDA llama ==")
    qwen = bench_qwen()
    print("== skip GLM remeasure (CapsWriter server holds the weights) ==")
    result = {
        "gpu": "RTX 5070 Ti, warm median, int4 ONNX + q5 decoder, CUDA EP + llama CUDA",
        "full_pipeline_ms": {
            "fun": {"5s": fun[5], "30s": fun[30]},
            "qwen": {"5s": qwen[5], "30s": qwen[30]},
        },
    }
    out = ROOT / "results/latency_full_cuda.json"
    out.write_text(__import__("json").dumps(result, indent=1, ensure_ascii=False))
    print("wrote", out)
    print(result)


if __name__ == "__main__":
    main()
