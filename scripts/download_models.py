#!/usr/bin/env python3
"""Download benchmark models from HuggingFace into models/ (relative layout).

GLM line:  zai-org base encoder exports + our trained CTC head (auto-fetched
           from the bench model repo, no manual quantization needed).
Fun line:  int4 trio auto-fetched from the same bench repo. Falls back to
           copying from a local CapsWriter-Offline install if offline.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, create_repo, upload_file

ROOT = Path(__file__).resolve().parent.parent
GLM_DIR = ROOT / "models" / "glm-ctc"
FUN_DIR = ROOT / "models" / "fun-asr"
QWEN_DIR = ROOT / "models" / "qwen-ctc"

HF_REPO = os.environ.get("BENCH_MODEL_REPO", "JazerJu/glm-asr-ctc-bench")

GLM_FILES = [
    # local source (publish once), remote name
    ("/data/推理框架/asr-onnx/GLM-ASR-CTC-GGUF/model/GLM-ASR-Encoder.q4.onnx", "GLM-ASR-Encoder.q4.onnx"),
    ("/data/推理框架/asr-onnx/GLM-ASR-CTC-GGUF/model/GLM-ASR-Encoder.q4.onnx.data", "GLM-ASR-Encoder.q4.onnx.data"),
    ("/data/推理框架/asr-onnx/GLM-ASR-CTC-GGUF/model/GLM-ASR-Projector.fp16.onnx", "GLM-ASR-Projector.fp16.onnx"),
    ("/data/推理框架/asr-onnx/GLM-ASR-CTC-GGUF/model/GLM-ASR-CTC-Final134k.q4.onnx", "GLM-ASR-CTC-Final134k.q4.onnx"),
    ("/data/推理框架/asr-onnx/GLM-ASR-CTC-GGUF/model/GLM-ASR-CTC-Final134k.fp32.onnx", "GLM-ASR-CTC-Final134k.fp32.onnx"),
    ("/data/推理框架/asr-onnx/GLM-ASR-CTC-GGUF/model/tokens-phase2.txt", "tokens-phase2.txt"),
]

# Fun-ASR-Nano int4 trio as packaged by CapsWriter-Offline (FunAudioLLM model,
# int4 conversion by HaujetZhao). Hosted here for one-stop reproducibility.
FUN_FILES = [
    "fun-asr/Fun-ASR-Nano-Encoder-Adaptor.int4.onnx",
    "fun-asr/Fun-ASR-Nano-CTC.int4.onnx",
    "fun-asr/tokens.txt",
]

QWEN_FILES = [
    "qwen-ctc/Qwen3-ASR-Encoder.q4.onnx",
    "qwen-ctc/Qwen3-ASR-CTC.q4.onnx",
    "qwen-ctc/tokens.txt",
    "qwen-ctc/preprocessor_config.json",
]


def publish():
    """One-time: push our exported/quantized GLM artifacts to the HF repo."""
    create_repo(HF_REPO, repo_type="model", exist_ok=True)
    for src, name in GLM_FILES:
        src = Path(src)
        if not src.exists():
            print(f"skip (missing): {src}")
            continue
        print(f"uploading {src.name} -> {HF_REPO}/{name} ({src.stat().st_size/1e6:.1f} MB)")
        upload_file(repo_id=HF_REPO, repo_type="model", path_or_fileobj=str(src), path_in_repo=name)
    print("publish done")


def fetch():
    GLM_DIR.mkdir(parents=True, exist_ok=True)
    for _, name in GLM_FILES:
        if name.endswith(".fp32.onnx") and not os.environ.get("FETCH_FP32"):
            continue  # 152 MB optional artifact, int4 is the default
        p = hf_hub_download(HF_REPO, name, repo_type="model")
        dst = GLM_DIR / name
        if not dst.exists():
            shutil.copyfile(p, dst)
        print(f"glm: {name} OK")


def link_fun():
    FUN_DIR.mkdir(parents=True, exist_ok=True)
    for name in FUN_FILES:
        dst = FUN_DIR / Path(name).name
        if dst.exists():
            print(f"fun: {dst.name} already present")
            continue
        p = hf_hub_download(HF_REPO, name, repo_type="model")
        shutil.copyfile(p, dst)
        print(f"fun: {dst.name} fetched from {HF_REPO}")


def fetch_qwen():
    QWEN_DIR.mkdir(parents=True, exist_ok=True)
    for name in QWEN_FILES:
        dst = QWEN_DIR / Path(name).name
        if dst.exists():
            print(f"qwen: {dst.name} already present")
            continue
        p = hf_hub_download(HF_REPO, name, repo_type="model")
        shutil.copyfile(p, dst)
        print(f"qwen: {dst.name} fetched from {HF_REPO}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fetch"
    if cmd == "publish":
        publish()
    elif cmd == "fetch":
        fetch()
        link_fun()
        fetch_qwen()
    elif cmd == "fun":
        link_fun()
    else:
        print(__doc__)
