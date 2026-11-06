# models/ layout

## glm-ctc/  (auto-fetched: `python scripts/download_models.py fetch`)
HF repo: `JazerJu/glm-asr-ctc-bench` — pre-quantized int4, no manual steps.
- GLM-ASR-Encoder.q4.onnx (+ .data external weights, keep both in this dir)
- GLM-ASR-Projector.fp16.onnx
- GLM-ASR-CTC-Final134k.q4.onnx (fp32 variant optional: FETCH_FP32=1)
- tokens-phase2.txt (BPE 59264)

## fun-asr/ (`python scripts/download_models.py fun`)
Third-party Fun-ASR-Nano int4 exports, mirrored in the same HF bench repo:
- Fun-ASR-Nano-Encoder-Adaptor.int4.onnx
- Fun-ASR-Nano-CTC.int4.onnx
- tokens.txt

## qwen-ctc/ (`python scripts/download_models.py fetch`)
Shared frozen encoder, original v1 head, base64 byte vocabulary, and
`preprocessor_config.json`. Do not substitute another vocabulary.

## qwen-ctc-a7avg5/ (`python scripts/download_models.py a7avg5`)
New head only: `Qwen3-ASR-CTC.{hqqf16,q4f16,q4}.onnx`, `config.json`.
`FETCH_FP16=1` also downloads `Qwen3-ASR-CTC.fp16.onnx`.
The engine `qwen_a7avg5` defaults to q4 and reuses `qwen-ctc/`; pass `--precision hqqf16` for the HQQ head.
For a7 DirectML, use `fetch-fp16` for the shared encoder and
`FETCH_FP16=1 python scripts/download_models.py a7avg5` for the new head.
See [release details](../docs/a7_avg5.md); all weights remain gitignored.
