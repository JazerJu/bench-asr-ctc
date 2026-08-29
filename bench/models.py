"""Model registry: unified engine loading for cross-model CTC benchmarking.

Each engine exposes the same interface:
    enc_output = engine.encode(audio_f32_16k)          # (T, D) encoder features
    text       = engine.decode_text(enc_output)         # greedy CTC first-pass

Engines: glm (GLM-ASR-Nano + our trained CTC), fun (Fun-ASR-Nano), qwen (reserved).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


def _ensure_cudnn():
    # onnxruntime-gpu needs cuDNN discoverable via LD_LIBRARY_PATH; common venv
    # locations are probed first, override with BENCH_CUDNN_LIB.
    candidates = [
        os.environ.get("BENCH_CUDNN_LIB", ""),
        "/data/模型训练/GlmAsr-Ctc训练/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib",
    ]
    for c in candidates:
        if c and Path(c).is_dir():
            os.environ["LD_LIBRARY_PATH"] = c + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
            return


def _ort_providers(use_gpu: bool):
    want = (os.environ.get("BENCH_ORT_PROVIDER") or "auto").lower()
    avail = set(ort.get_available_providers())
    if (not use_gpu) or want == "cpu":
        return ["CPUExecutionProvider"]
    if want in ("auto", "cuda") and "CUDAExecutionProvider" in avail:
        _ensure_cudnn()
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if want in ("auto", "dml") and "DmlExecutionProvider" in avail:
        return ["DmlExecutionProvider", "CPUExecutionProvider"]
    if want == "cuda":
        print("[warn] CUDA EP not installed, using CPU")
    elif want == "dml":
        print("[warn] DML EP not installed, using CPU")
    return ["CPUExecutionProvider"]


def _precision_tag(quantized: bool = True) -> str:
    tag = (os.environ.get("BENCH_PRECISION") or "").lower()
    if tag in ("q4", "fp16", "fp32"):
        return tag
    return "q4" if quantized else "fp32"


def _session(onnx_path: Path, use_gpu: bool, intra_threads: int = 6):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = _ort_providers(use_gpu)
    if providers[0] == "CPUExecutionProvider":
        opts.intra_op_num_threads = intra_threads
    try:
        return ort.InferenceSession(str(onnx_path), sess_options=opts, providers=providers)
    except Exception:
        if providers[0] != "CPUExecutionProvider":
            print(f"[warn] {providers[0]} failed for {onnx_path.name}, falling back to CPU")
            return ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
        raise


def _load_tokens(path: Path, fun_style: bool = False) -> dict:
    id2token = {}
    import base64
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if fun_style:
                parts = line.split()
                t, i = (" ", parts[0]) if len(parts) == 1 else parts
                try:
                    tok = base64.b64decode(t).decode("utf-8")
                except Exception:
                    tok = t
                id2token[int(i)] = tok
            else:
                tok, idx = line.rsplit("\t", 1)
                id2token[int(idx)] = json.loads(tok)
    return id2token


def _greedy_collapse(argmax_ids, blank_id, id2token):
    prev = None
    out = []
    for tid in argmax_ids:
        tid = int(tid)
        if tid == blank_id:
            prev = None
            continue
        if tid == prev:
            continue
        out.append(id2token.get(tid, ""))
        prev = tid
    return "".join(out)


class GLMEngine:
    """GLM-ASR-Nano encoder + our trained CTC head (final_134k, int4 by default)."""

    def __init__(self, use_gpu: bool = True, quantized: bool = True):
        glm_dir = MODELS_DIR / "glm-ctc"
        tag = _precision_tag(quantized)
        enc_path = glm_dir / f"GLM-ASR-Encoder.{tag}.onnx"
        enc_data = glm_dir / f"GLM-ASR-Encoder.{tag}.onnx.data"
        proj_path = glm_dir / "GLM-ASR-Projector.fp16.onnx"
        ctc_path = glm_dir / f"GLM-ASR-CTC-Final134k.{tag}.onnx"
        for p in (enc_path, proj_path, ctc_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"{p} not found. Run: python scripts/download_models.py  "
                    f"(or place files manually, see models/README.md)"
                )
        _ = enc_data  # external weights must sit next to encoder.q4.onnx

        self.enc_sess = _session(enc_path, use_gpu)
        self.enc_input = self.enc_sess.get_inputs()[0].name
        self.proj_sess = _session(proj_path, use_gpu)
        self.proj_input = self.proj_sess.get_inputs()[0].name

        self.ctc_sess = _session(ctc_path, use_gpu)
        self.ctc_input = self.ctc_sess.get_inputs()[0].name
        self.blank_id = 59263
        self.id2token = _load_tokens(glm_dir / "tokens-phase2.txt")

        from transformers import AutoProcessor
        model_id = os.environ.get("GLM_ASR_MODEL_ID", "zai-org/GLM-ASR-Nano-2512")
        snaps = sorted(Path("/data/.cache/huggingface/hub/models--zai-org--GLM-ASR-Nano-2512/snapshots").glob("*"))
        load_id = str(snaps[-1]) if snaps else model_id
        # AutoProcessor 在新版 transformers 上对这个仓库直接返回
        # WhisperFeatureExtractor 本身（快照里只有 processor_config.json，
        # 没有 preprocessor_config.json），再取 .feature_extractor 就
        # AttributeError。两种返回都接住。
        proc = AutoProcessor.from_pretrained(
            load_id, trust_remote_code=True, local_files_only=bool(snaps)
        )
        self.fe = getattr(proc, "feature_extractor", proc)

    def _mel(self, audio: np.ndarray) -> np.ndarray:
        if self.fe is not None:
            import numpy.lib.format as _fmt  # noqa
            extracted = self.fe(audio.astype(np.float32), sampling_rate=16000, return_tensors="np").input_features
            full = np.asarray(extracted, dtype=np.float32)
            hop = self.fe.hop_length
            n_expected = int(len(audio) / hop) + 1
            n_trim = min(n_expected, full.shape[2])
            pad_to = ((n_trim + 7) // 8) * 8
            if pad_to > full.shape[2]:
                trimmed = np.pad(full[:, :, :n_trim], ((0, 0), (0, 0), (0, pad_to - n_trim)))
            else:
                trimmed = full[:, :, :pad_to]
            return trimmed
        import librosa
        window = np.hanning(401)[:-1]
        stft = np.abs(librosa.stft(audio, n_fft=400, hop_length=160, window=window, center=True, pad_mode="reflect")) ** 2
        mel = librosa.filters.mel(sr=16000, n_fft=400, n_mels=128) @ stft
        mel = np.log10(np.maximum(mel, 1e-10))
        mel = np.maximum(mel - 1.0, -1.0) / 4.0
        n_frames = mel.shape[1]
        pad_to = ((n_frames + 7) // 8) * 8
        if pad_to > n_frames:
            mel = np.pad(mel, ((0, 0), (0, pad_to - n_frames)))
        return mel[np.newaxis, ...]

    def encode(self, audio: np.ndarray):
        features = self._mel(audio).astype(np.float32)
        enc_out = self.enc_sess.run(None, {self.enc_input: features})[0]
        enc_output = enc_out[np.newaxis, ...] if enc_out.ndim == 2 else enc_out
        _ = self.proj_sess  # projector kept for pipeline parity; CTC bench only needs enc_output
        return enc_output

    def decode_text(self, enc_output: np.ndarray) -> str:
        logits = self.ctc_sess.run(None, {self.ctc_input: enc_output.astype(np.float32)})[0]
        logits = logits[0] if logits.ndim == 3 else logits
        argmax_ids = np.argmax(logits, axis=-1)
        return _greedy_collapse(argmax_ids, self.blank_id, self.id2token)


class _FunMelExtractor:
    """torchaudio-exact mel frontend (htk scale, periodic hamming, zero-pad STFT)."""

    def __init__(self, sr=16000, n_fft=400, n_mels=80, f_min=20, f_max=8000):
        hz_to_mel = lambda f: 2595.0 * np.log10(1.0 + (f / 700.0))
        mel_to_hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
        all_freqs = np.linspace(0, sr // 2, n_fft // 2 + 1)
        m_pts = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
        f_pts = mel_to_hz(m_pts)
        f_diff = np.diff(f_pts)
        slopes = f_pts[np.newaxis, :] - all_freqs[:, np.newaxis]
        down = (-1.0 * slopes[:, :-2]) / f_diff[:-1]
        up = slopes[:, 2:] / f_diff[1:]
        self.filters = np.maximum(0, np.minimum(down, up)).astype(np.float32)
        self.hop = 160
        self.n_fft = n_fft
        self.window = (0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(n_fft) / n_fft)).astype(np.float32)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        audio = audio - np.mean(audio)
        pe = np.empty_like(audio)
        pe[0] = audio[0]
        pe[1:] = audio[1:] - 0.97 * audio[:-1]
        y = np.pad(pe, (self.n_fft // 2, self.n_fft // 2), mode="constant")
        num_frames = 1 + (len(y) - self.n_fft) // self.hop
        frames = np.lib.stride_tricks.as_strided(
            y, shape=(num_frames, self.n_fft), strides=(y.strides[0] * self.hop, y.strides[0])
        )
        mag = np.abs(np.fft.rfft(frames * self.window, n=self.n_fft, axis=1)) ** 2
        return np.log(mag @ self.filters + 1e-7)  # (T_mel, 80)


class FunEngine:
    """Fun-ASR-Nano (SenseVoice encoder + 5-block CTC decoder), int4."""

    _mel = None

    def __init__(self, use_gpu: bool = False):
        fun_dir = MODELS_DIR / "fun-asr"
        enc_path = fun_dir / "Fun-ASR-Nano-Encoder-Adaptor.int4.onnx"
        ctc_path = fun_dir / "Fun-ASR-Nano-CTC.int4.onnx"
        tokens_path = fun_dir / "tokens.txt"
        for p in (enc_path, ctc_path, tokens_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"{p} not found. Copy from a CapsWriter-Offline installation "
                    f"(models/Fun-ASR-Nano/Fun-ASR-Nano-GGUF/), see models/README.md"
                )
        self.enc_sess = _session(enc_path, use_gpu=use_gpu)
        self.enc_input = self.enc_sess.get_inputs()[0].name
        self.ctc_sess = _session(ctc_path, use_gpu=False)
        self.ctc_input = self.ctc_sess.get_inputs()[0].name
        id2token = _load_tokens(tokens_path, fun_style=True)
        self.blank_id = max(id2token.keys())
        self.id2token = id2token
        if FunEngine._mel is None:
            FunEngine._mel = _FunMelExtractor()

    @staticmethod
    def _lfr(audio: np.ndarray) -> np.ndarray:
        # Fun-ASR frontend: torchaudio-exact 80-mel -> LFR 7-frame stack, stride 6
        log_mel = FunEngine._mel(audio)  # (T_mel, 80)

        t_mel = log_mel.shape[0]
        t_lfr = (t_mel + 5) // 6
        left = np.repeat(log_mel[:1], 3, axis=0)
        right_len = (t_lfr * 6 + 7) - t_mel
        right = np.repeat(log_mel[-1:], max(right_len, 0), axis=0)
        padded = np.concatenate([left, log_mel, right], axis=0)

        lfr_feat = np.empty((t_lfr, 560), dtype=np.float32)
        for i in range(7):
            lfr_feat[:, i * 80 : (i + 1) * 80] = padded[i : i + t_lfr * 6 : 6, :]
        return lfr_feat[np.newaxis, ...]

    def encode(self, audio: np.ndarray):
        feats = self._lfr(audio)
        mask = np.ones((1, feats.shape[1]), dtype=np.float32)
        outputs = self.enc_sess.run(None, {self.enc_input: feats, self.enc_sess.get_inputs()[1].name: mask})
        return outputs[0]

    def decode_text(self, enc_output: np.ndarray) -> str:
        logits = self.ctc_sess.run(None, {self.ctc_input: enc_output.astype(np.float32)})[0]
        if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[0] == 1):
            indices = logits.flatten()
        else:
            indices = np.argmax(logits.astype(np.float32), axis=-1).flatten()
        return _greedy_collapse(indices, self.blank_id, self.id2token)


class QwenEngine:
    """Qwen3-ASR encoder (frozen official weights) + our trained CTC head.

    13 fps encoder output, byte-level BPE detokenization (see
    QWEN3-CTC-导出与推理指示.md §1.1/§1.4); fixed 30s ONNX bucket with a
    runtime feature_length input, output sliced by the export-safe length
    formula. Padded frames are attention-masked inside the graph.
    """

    BLANK_ID = 72466
    UNK_ID = 72467
    MEL_FRAMES = 3000

    def __init__(self, use_gpu: bool = True, quantized: bool = True):
        d = MODELS_DIR / "qwen-ctc"
        tag = _precision_tag(quantized)
        self.enc_sess = _session(d / f"Qwen3-ASR-Encoder.{tag}.onnx", use_gpu=use_gpu)
        self.ctc_sess = _session(d / f"Qwen3-ASR-CTC.{tag}.onnx", use_gpu=use_gpu)
        self.id2bytes = {}
        import base64

        for line in open(d / "tokens.txt", encoding="utf-8"):
            line = line.rstrip("\n")
            if not line:
                continue
            b64, idx = line.rsplit("\t", 1)
            self.id2bytes[int(idx)] = base64.b64decode(b64)
        from transformers import WhisperFeatureExtractor

        self.fe = WhisperFeatureExtractor.from_pretrained(str(d))

    def encode(self, audio: np.ndarray):
        mel = self.fe(audio, sampling_rate=16000, padding=False,
                      return_tensors="np").input_features[0]
        T = mel.shape[1]
        padded = np.zeros((128, self.MEL_FRAMES), dtype=np.float32)
        padded[:, :T] = mel
        out = self.enc_sess.run(None, {"input_features": padded,
                                       "feature_length": np.array([T], np.int64)})[0]
        full, leave = divmod(T, 100)
        tail = 0 if leave == 0 else (leave - 1) // 8 + 1
        return out[: full * 13 + tail]

    def decode_text(self, enc_output: np.ndarray) -> str:
        x = enc_output[None] if enc_output.ndim == 2 else enc_output
        logits = self.ctc_sess.run(None, {"enc_output": x.astype(np.float32)})[0]
        ids = np.argmax(logits[0], axis=-1)
        kept, prev = [], None
        for t in ids:
            t = int(t)
            if t == prev:
                continue
            prev = t
            if t not in (self.BLANK_ID, self.UNK_ID):
                kept.append(t)
        return b"".join(self.id2bytes[t] for t in kept).decode("utf-8", errors="replace")


class QwenR2Engine(QwenEngine):
    """Qwen3-ASR-CTC r2: same frozen encoder, wider FFN CTC head (ffn_hidden=2048)."""

    def __init__(self, use_gpu: bool = True, quantized: bool = True):
        d_enc = MODELS_DIR / "qwen-ctc"
        d_ctc = MODELS_DIR / "qwen-ctc-r2"
        tag = "q4" if quantized else "fp32"
        self.enc_sess = _session(d_enc / f"Qwen3-ASR-Encoder.{tag}.onnx", use_gpu=use_gpu)
        ctc_path = d_ctc / f"Qwen3-ASR-CTC.{tag}.onnx"
        if not ctc_path.exists():
            ctc_path = d_ctc / "Qwen3-ASR-CTC.q4.onnx"
        self.ctc_sess = _session(ctc_path, use_gpu=use_gpu)
        self.id2bytes = {}
        import base64
        for line in open(d_enc / "tokens.txt", encoding="utf-8"):
            line = line.rstrip("\n")
            if not line:
                continue
            b64, idx = line.rsplit("\t", 1)
            self.id2bytes[int(idx)] = base64.b64decode(b64)
        from transformers import WhisperFeatureExtractor
        self.fe = WhisperFeatureExtractor.from_pretrained(str(d_enc))


class QwenR2Fp16Engine(QwenR2Engine):
    def __init__(self, use_gpu: bool = True, quantized: bool = True):
        d_enc = MODELS_DIR / "qwen-ctc"
        d_ctc = MODELS_DIR / "qwen-ctc-r2"
        self.enc_sess = _session(d_enc / "Qwen3-ASR-Encoder.q4.onnx", use_gpu=use_gpu)
        self.ctc_sess = _session(d_ctc / "Qwen3-ASR-CTC.fp16.onnx", use_gpu=use_gpu)
        self.id2bytes = {}
        import base64
        for line in open(d_enc / "tokens.txt", encoding="utf-8"):
            line = line.rstrip("\n")
            if not line:
                continue
            b64, idx = line.rsplit("\t", 1)
            self.id2bytes[int(idx)] = base64.b64decode(b64)
        from transformers import WhisperFeatureExtractor
        self.fe = WhisperFeatureExtractor.from_pretrained(str(d_enc))


ENGINES = {
    "glm": GLMEngine,
    "fun": FunEngine,
    "qwen": QwenEngine,
    "qwen_r2": QwenR2Engine,
    "qwen_r2_fp16": QwenR2Fp16Engine,
}


def get_engine(name: str, **kwargs):
    if name not in ENGINES:
        raise KeyError(f"unknown engine {name!r}, available: {list(ENGINES)}")
    return ENGINES[name](**kwargs)
