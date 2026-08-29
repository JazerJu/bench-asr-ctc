# AGENTS.md — bench-asr-ctc

## What This Repo Does

Cross-model CTC **first-pass** benchmark: three engines behind one interface,
same audio, same metrics. No LLM second pass — this measures the CTC head only.

| Engine | Encoder | CTC head | Repo |
|---|---|---|---|
| `glm` | GLM-ASR-Nano (635 M, 50 fps) | ours, 59,264 classes | [GLM-ASR-CTC-GGUF](https://github.com/JazerJu/GLM-ASR-CTC-GGUF) |
| `qwen` | Qwen3-ASR-1.7B (317.5 M, 13 fps) | ours, 72,468 classes | [Qwen3-ASR-CTC-GGUF](https://github.com/JazerJu/Qwen3-ASR-CTC-GGUF) |
| `fun` | Fun-ASR-Nano | upstream | [Fun-ASR-GGUF](https://github.com/HaujetZhao/Fun-ASR-GGUF) |

Not an export repo — the Fun-ASR-GGUF numbered-pipeline convention does not
apply here. Exports come from the two repos above.

## Engine Contract

Every engine in `bench/models.py` implements exactly two methods:

```python
class SomeEngine:
    def __init__(self, use_gpu: bool = True, quantized: bool = True): ...
    def encode(self, audio: np.ndarray): ...        # -> encoder output
    def decode_text(self, enc_output) -> str: ...   # -> greedy transcript
```

Register it in `ENGINES` and every runner works unchanged.

## Layout

```
bench.py              multi-language FLEURS driver
infer.py              single-audio CLI, any engine
bench/models.py       engine registry: GLMEngine / FunEngine / QwenEngine
bench/metrics.py      WER / CER / MER + per-language normalization
bench/hotword/        pypinyin phoneme DP matching (CapsWriter-derived)
runners/              bench_fleurs.py, bench_hotword.py, bench_buckeye.py, parity_check.py
cases/                4 bundled 30s clips + expected hotwords
models/<engine>/      ONNX + tokens (gitignored; fetched by scripts/download_models.py)
results/              result JSONs, each with a `meta` block
```

## Invariants — do not "fix" these

1. **Metric per language.** CER for zh/ja/ko (chars after whitespace removal),
   WER for latin (whitespace tokens), MER for code-switch (CJK by char, latin
   runs by word). Plain CER splits English words into characters; plain WER
   treats a whole Chinese sentence as one token.
2. **Same execution provider on both sides.** Fun's int4 ONNX is *not*
   provider-invariant — ORT CPU vs CUDA EP transcripts differ on ~5% of samples
   (up to 1.9 pp CER, CPU weaker). Every published number must state its EP, and
   comparisons must use the same one. `runners/parity_check.py` measures this.
3. **Qwen3 is 13 fps, not 50.** `valid = full*13 + ceil(leave/8)` where
   `full, leave = divmod(mel_frames, 100)`. Also: `padding=False`, then pad to
   the 3000-frame bucket and pass the real `feature_length`.
4. **Qwen3 detokenizes through bytes.** `tokens.txt` holds base64 raw bytes per
   compact id; join bytes then decode once. `"".join(pieces)` yields mojibake.
5. **GLM's feature extractor**: `AutoProcessor.from_pretrained` may return a
   bare `WhisperFeatureExtractor` on newer transformers (the snapshot has
   `processor_config.json` but no `preprocessor_config.json`). Use
   `getattr(proc, "feature_extractor", proc)`.
6. **Sampling is deterministic** — evenly strided over the split, so identical
   `--counts` always evaluates identical sentences. Do not shuffle.

## Before claiming a run is done

`python infer.py cases/zh_news_30s.wav --engine <e> --cpu` must print a
**non-empty** transcript for every engine you touched. An empty transcript is
the signature of a feature-normalization or frame-rate bug and raises nothing.
