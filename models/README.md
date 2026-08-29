# models/ layout

## glm-ctc/  (auto-fetched: `python scripts/download_models.py fetch`)
HF repo: `JazerJu/glm-asr-ctc-bench` — pre-quantized int4, no manual steps.
- GLM-ASR-Encoder.q4.onnx (+ .data external weights, keep both in this dir)
- GLM-ASR-Projector.fp16.onnx
- GLM-ASR-CTC-Final134k.q4.onnx (fp32 variant optional: FETCH_FP32=1)
- tokens-phase2.txt (BPE 59264)

## fun-asr/  (manual copy: `python scripts/download_models.py fun`)
Third-party Fun-ASR-Nano, not redistributed. Copy these 3 files from a
CapsWriter-Offline install (`models/Fun-ASR-Nano/Fun-ASR-Nano-GGUF/`):
- Fun-ASR-Nano-Encoder-Adaptor.int4.onnx
- Fun-ASR-Nano-CTC.int4.onnx
- tokens.txt

## qwen-ctc/  (RESERVED)
Drop future Qwen3-ASR CTC ONNX exports here, then implement
`QwenEngine.encode/decode_text` in bench/models.py (interface documented in-code).
