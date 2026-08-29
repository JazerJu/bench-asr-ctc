# bench-asr-ctc

Cross-model CTC first-pass benchmark: **GLM-ASR (our trained CTC head) vs Fun-ASR-Nano vs Qwen3-ASR (our trained CTC head)** — three engines, one interface.

Tests three capabilities on identical audio:
1. **Transcription quality** — FLEURS official test splits (11 languages), WER/CER
2. **Hotword DP matching** — CapsWriter-style pypinyin phoneme + fuzzy edit-distance pipeline (CTC text is never modified; matched hotwords are reported as LLM-prompt hints)
3. **Reproducibility** — every engine behind one interface (`encode` → `decode_text`), int4-on-int4 fair comparison

## Results snapshot (all engines int4, FLEURS **full official test split**, 7,876 samples, all on CUDA EP)

| lang | metric | n | GLM-CTC | Fun-ASR-Nano | Qwen3-CTC | winner |
|---|---|---|---|---|---|---|
| en_us | WER | 647 | 18.3% | **18.1%** | 18.3% | Fun (tie, 0.2pp) |
| cmn_hans_cn | CER | 945 | 10.7% | **9.3%** | 10.9% | Fun |
| ko_kr | CER | 382 | 39.5% | 46.0% | **20.3%** | **Qwen (2× better)** |
| ja_jp | CER | 650 | 31.0% | **17.0%** | 30.9% | Fun |
| yue_hant_hk | CER | 819 | 42.7% | 43.2% | **30.1%** | **Qwen** |
| de_de | WER | 862 | **35.7%** | 75.5% | 41.6% | GLM |
| fr_fr | WER | 676 | **39.8%** | 76.6% | 46.6% | GLM |
| es_419 | WER | 908 | **29.7%** | 63.0% | 35.6% | GLM |
| it_it | WER | 865 | **40.8%** | 86.2% | 50.9% | GLM |
| nl_nl | WER | 364 | **46.7%** | 95.8% | 50.1% | GLM |
| pl_pl | WER | 758 | **69.3%** | 98.8% | 77.1% | GLM |

**GLM 7W / Fun 2W / Qwen 2W.** Raw: `results/fleurs_full_3way.json`. Buckeye conversational English (full, 2,477 segs / 20 speakers / 145,690 words, 1 empty skipped): **Qwen 18.1% < GLM 22.4% < Fun 26.9%** (`results/buckeye_full.json`). Spontaneous English still favors Qwen; GLM is clearly ahead of Fun. The earlier 40-seg s01-only slice (Qwen 23.3 / GLM 36.2 / Fun 41.4) was a hard speaker, not the corpus. Read speech (FLEURS en) remains a three-way tie.

**Chinese & zh/en-mixed** (FLEURS sub-cuts, `runners/bench_mixed.py`, 3-way): Mandarin pure CER Fun 8.4 < GLM 9.7 ≈ Qwen 9.8; Mandarin mixed-CER Fun 17.0 < GLM 19.2 ≈ Qwen 19.1 — Fun keeps a small edge on wiki-style read speech. Cantonese (incl. embedded English): Qwen 29.6 pure / 37.7 mixed — 12pp+ ahead of both. Spoken tech-term recall (fq video, 37 windows): Fun 54% > Qwen 32% > GLM 27%, with **complementary misses** — Qwen catches bbr/gfw/cpu that GLM drops, GLM catches ssl that Qwen drops (`results/spoken_term_recall_3way.json`).

Qwen3-CTC notes: encoder is half of GLM's params (317.5M vs 635.0M) at 13fps (GLM: 50fps) — roughly 1/8 the compute per audio second. Its ko/yue wins come from the Qwen3-ASR base's multilingual pretraining; the GLM-vs-Qwen gap on zh/en traces to training recipe (same data, 512×56,916 vs 256×134,140 updates — controlled experiment pending).

Provider parity note: Fun int4 ONNX is **not** provider-invariant — ORT CPU vs CUDA EP transcripts differ on ~5% of samples (up to 1.9pp CER, CPU weaker; see `runners/parity_check.py`). All published numbers use all engines on CUDA EP.

Reproducibility: `python bench.py --counts 200 --engines glm fun qwen` (~2,200 samples/engine, deterministic strided sampling — identical `--counts` always evaluates identical sentences).

## Setup

```bash
pip install -r requirements.txt          # onnxruntime-gpu optional

# Everything auto-fetched from HuggingFace (GLM line + Fun-ASR-Nano int4 trio, ~600 MB):
python scripts/download_models.py fetch
#    = glm-ctc/  from JazerJu/glm-asr-ctc-bench (encoder q4, projector, CTC final134k q4, tokens)
#    + fun-asr/  from JazerJu/glm-asr-ctc-bench/fun-asr (int4 trio; FunAudioLLM model,
#                int4 conversion as packaged by CapsWriter-Offline)
```

## Run

```bash
# sanity: 2 langs × 20 sentences (~30s)
python bench.py --counts 20 --langs en_us,cmn_hans_cn

# issue-table scale: 11 langs × 200 sentences (~20 min on GPU)
python bench.py --counts 200

# full official test split (7,876 samples)
python bench.py --counts full

# single-audio inference, either backend (CTC first-pass only, any ffmpeg input, ≤30s)
python infer.py clip.wav --engine glm
python infer.py input.mp3 --engine fun --cpu

# hotword DP matching on bundled cases (4 × 30s, expected hits encoded)
python runners/bench_hotword.py --engines glm fun

# fp32 parity check (GLM only)
python bench.py --counts 20 --engines glm --fp32 --langs en_us
```

Every result JSON records a `meta` block (engine revisions, quantization, ORT version, execution providers, timestamp). FLEURS parquet shards are fetched automatically via `huggingface_hub` and cached under `HF_HOME` — no manual dataset download.

## Layout

```
bench/
  models.py        # engine registry: GLMEngine / FunEngine / QwenEngine(slot)
  metrics.py       # WER/CER + per-language normalization
  hotword/         # PhonemeCorrector DP matching (CapsWriter-derived)
runners/
  bench_fleurs.py  # multi-language transcription benchmark
  bench_hotword.py # hotword hit/miss/false-positive check
cases/             # 4 bundled 30s test clips + expected hotwords
models/
  glm-ctc/         # encoder.q4.onnx(+.data), projector, ctc-final134k.q4.onnx, tokens
  fun-asr/         # Fun-ASR-Nano int4 trio (from JazerJu/glm-asr-ctc-bench/fun-asr)
  qwen-ctc/        # Qwen3-ASR encoder q4, our CTC head q4/fp32, tokens, preprocessor
scripts/
  download_models.py  # fetch (HF) / fun (local copy) / publish (one-time upload)
```

## The Qwen engine

`QwenEngine` in `bench/models.py` is live: frozen official Qwen3-ASR encoder (int4, 13fps fixed-30s-bucket ONNX) + our trained CTC head (`JazerJu/qwen3-asr-ctc`, 56,916 steps). ONNX-chain parity vs the training-side PyTorch: LibriSpeech test-clean WER 6.92% vs 6.93% (`runners/qwen_librispeech_parity.py`). Export details: `../Qwen3-ASR-CTC-GGUF/`.

## Model provenance

- GLM encoder/projector: exported from `zai-org/GLM-ASR-Nano-2512` (MIT), int4-quantized by us
- GLM CTC head: our training (134k steps, warmup + 5 transformer blocks, BPE 59264), published at `JazerJu/glm-asr-ctc-bench`
- Fun-ASR-Nano: HaujetZhao/CapsWriter-Offline model suite (int4 trio mirrored at JazerJu/glm-asr-ctc-bench/fun-asr)
- Qwen3 encoder: official `Qwen/Qwen3-ASR-1.7B` audio tower (frozen), int4 by us; CTC head ours (`JazerJu/qwen3-asr-ctc`)
- FLEURS: `google/fleurs` (CC-BY-4.0), auto-fetched
