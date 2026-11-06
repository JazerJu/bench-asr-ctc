# bench-asr-ctc

[English](README.md) · [中文](README_zh.md)

Cross-model CTC first-pass benchmark: **GLM-ASR (our trained CTC head) vs Fun-ASR-Nano vs Qwen3-ASR (our trained CTC head)** — three engines, one interface.

Tests three capabilities on identical audio:
1. **Transcription quality** — FLEURS official test splits (11 languages), WER/CER
2. **Hotword DP matching** — CapsWriter-style pypinyin phoneme + fuzzy edit-distance pipeline (CTC text is never modified; matched hotwords are reported as LLM-prompt hints)
3. **Reproducibility** — every engine behind one interface (`encode` → `decode_text`), int4-on-int4 fair comparison

## a7_avg5 Update

The ONNX and PyTorch weights, model cards, and checksums are published on HF.
The a7 download below is opt-in; existing `fetch` remains unchanged.

The new self-conditioned Qwen3 CTC head is registered as `qwen_a7avg5` (default: q4, same as other rounds; HQQ via `--precision hqqf16`). It reduces the English-to-Chinese leakage
reported in [CapsWriter #457](https://github.com/HaujetZhao/CapsWriter-Offline/issues/457).
It does not replace the historical `qwen` or `qwen_aiinfra4` weights.

```bash
python scripts/download_models.py a7avg5
python bench.py --counts 20 --engines qwen_a7avg5 --langs en_us,cmn_hans_cn
python scripts/summarize_a7.py  # saved a6/a7 results; no audio, weights, or GPU
```

On the 17-clip Chinese regression subset, a7_avg5 HQQ CER is **14.96%**
(a6_avg5 HQQ: 16.20%); the 6-clip English regression CER is **5.98%** (6.68%).
References are v1+LLM pseudo-labels and these are per-clip macro means, not
held-out corpus scores. TED leakage is reduced but not eliminated (6 CTC Han
characters in 29 talks); TED overlaps training data.

[Release details, precision choices, raw results, and audio preparation](docs/a7_avg5.md).
Long-video audio is not bundled: fresh inference needs the source media, while
saved-result summaries do not. The tables below retain their original revisions;
they are **not** re-labelled as a7_avg5 results.

## Historical Results (all engines int4, FLEURS **full official test split**, 7,876 samples, all on CUDA EP)

> These are **v1-era** numbers (2026-08/09): Qwen3-CTC is the 56,916-step v1 head and GLM is
> `final134k`. They are not a7_avg5 results — current head: [docs/a7_avg5.md](docs/a7_avg5.md).

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

**GLM 6W / Fun 3W / Qwen 2W.** Raw: `results/fleurs_full_3way.json`. Buckeye conversational English (full, 2,477 segs / 20 speakers / 145,690 words, 1 empty skipped): **Qwen 18.1% < GLM 22.4% < Fun 26.9%** (`results/buckeye_full.json`). Spontaneous English still favors Qwen; GLM is clearly ahead of Fun. The earlier 40-seg s01-only slice (Qwen 23.3 / GLM 36.2 / Fun 41.4) was a hard speaker, not the corpus. Read speech (FLEURS en) remains a three-way tie.

**Chinese & zh/en-mixed** (FLEURS sub-cuts, `runners/bench_mixed.py`, 3-way): Mandarin pure CER Fun 8.4 < GLM 9.7 ≈ Qwen 9.8; Mandarin mixed-CER Fun 17.0 < GLM 19.2 ≈ Qwen 19.1 — Fun keeps a small edge on wiki-style read speech. Cantonese (incl. embedded English): Qwen 29.6 pure / 37.7 mixed — 12pp+ ahead of both. Spoken tech-term recall (fq video, 37 windows): Fun 54% > Qwen 32% > GLM 27%, with **complementary misses** — Qwen catches bbr/gfw/cpu that GLM drops, GLM catches ssl that Qwen drops (`results/spoken_term_recall_3way.json`).

Qwen3-CTC notes: encoder is half of GLM's params (317.5M vs 635.0M) at 13fps (GLM: 50fps) — roughly 1/8 the compute per audio second. Its ko/yue wins come from the Qwen3-ASR base's multilingual pretraining; the GLM-vs-Qwen gap on zh/en traces to training recipe (same data, 512×56,916 vs 256×134,140 updates — controlled experiment pending).

Provider parity note: Fun int4 ONNX is **not** provider-invariant — ORT CPU vs CUDA EP transcripts differ on ~5% of samples (up to 1.9pp CER, CPU weaker; see `runners/parity_check.py`). All published numbers use all engines on CUDA EP. a7_avg5's RTN q4/q4f16 and HQQ hqqf16 heads
are also MatMulNBits graphs, so DirectML needs the fp16 graphs (`--precision fp16`).

Reproducibility: `python bench.py --counts 200 --engines glm fun qwen` (~2,200 samples/engine, deterministic strided sampling — identical `--counts` always evaluates identical sentences).

## Setup

```bash
pip install -r requirements.txt          # onnxruntime-gpu optional

# Linux / NVIDIA CUDA (int4, default):
python scripts/download_models.py fetch

# Windows iGPU DirectML (fp16, no MatMulNBits):
#   pip install onnxruntime-directml
#   python scripts/download_models.py fetch-fp16
#   python bench.py --counts 20 --provider dml --precision fp16 --engines glm,qwen
#    = glm-ctc/  from JazerJu/glm-asr-ctc-bench (encoder q4, projector, CTC final134k q4, tokens)
#    + fun-asr/  from JazerJu/glm-asr-ctc-bench/fun-asr (int4 trio; FunAudioLLM model,
#                int4 conversion as packaged by CapsWriter-Offline)
#    + qwen-ctc/ from JazerJu/glm-asr-ctc-bench/qwen-ctc (encoder q4 + CTC q4 +
#                tokens.txt + preprocessor_config.json)
#
# One command pulls the three engines plus the aiinfra4 head. a7_avg5 is opt-in:
#   python scripts/download_models.py a7avg5
# FETCH_FP16=1 adds the fp16 variants, FETCH_FP32=1 adds GLM's fp32. Everything
# under models/ comes from this step and is gitignored — the repo tracks no weights.
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
  models.py        # engine registry: GLMEngine / FunEngine / QwenEngine (+aiinfra rounds)
  metrics.py       # WER/CER + per-language normalization
  hotword/         # PhonemeCorrector DP matching (CapsWriter-derived)
runners/
  bench_fleurs.py  # multi-language transcription benchmark
  bench_hotword.py # hotword hit/miss/false-positive check
cases/             # 4 bundled 30s clips (cases.json + hotwords) and the long-video sets:
                   #   en_regression/ zh_regression/ zh_long20/ ted_collection/
                   #   tedx_fukuoka/ ascend_cube/  (audio is not tracked; see each README)
docs/
  a7_avg5.md       # current head: results, precision, audio availability
tests/
  test_release.py  # engine paths, downloader layout, audio preparation
models/
  glm-ctc/         # encoder.q4.onnx(+.data), projector, ctc-final134k.q4.onnx, tokens
  fun-asr/         # Fun-ASR-Nano int4 trio (from JazerJu/glm-asr-ctc-bench/fun-asr)
  qwen-ctc/        # shared encoder q4, v1 head, tokens, preprocessor
  qwen-ctc-aiinfra4/  # aiinfra4 head (q4/q4f16) — download_models.py aiinfra4
  qwen-ctc-a7avg5/    # a7_avg5 head (q4/q4f16/hqqf16) — download_models.py a7avg5
scripts/
  download_models.py  # fetch / aiinfra4 / a7avg5 / fetch-fp16 / fun / publish
  en_regression.py    # long-audio regression runner (CapsWriter runtime)
  ted_collection_check.py / summarize_a7.py / prepare_case_audio.py / del_pos.py
```

## The Qwen engine

`QwenEngine` (`qwen`) is v1: frozen official Qwen3-ASR encoder (int4, 13fps fixed-30s-bucket
ONNX) + our 56,916-step head (`JazerJu/qwen3-asr-ctc`). ONNX-chain parity vs the training-side
PyTorch: LibriSpeech test-clean WER 6.92% vs 6.93% (`runners/qwen_librispeech_parity.py`).

Later rounds reuse the same frozen encoder and vocabulary, swapping only the head:
`qwen_aiinfra4` (2026-09) and `qwen_a7avg5` (current, fixes the English-audio Chinese leakage
from #457). See [docs/a7_avg5.md](docs/a7_avg5.md) for results and caveats.
Export details: `../Qwen3-ASR-CTC-GGUF/`.

## Model provenance

- GLM encoder/projector: exported from `zai-org/GLM-ASR-Nano-2512` (MIT), int4-quantized by us
- GLM CTC head: our training (134k steps, warmup + 5 transformer blocks, BPE 59264), published at `JazerJu/glm-asr-ctc-bench`
- Fun-ASR-Nano: HaujetZhao/CapsWriter-Offline model suite (int4 trio mirrored at JazerJu/glm-asr-ctc-bench/fun-asr)
- Qwen3 encoder: official `Qwen/Qwen3-ASR-1.7B` audio tower (frozen), int4 by us
- Qwen3 CTC heads, all ours: v1 `JazerJu/qwen3-asr-ctc`; aiinfra4 and a7_avg5 under
  `JazerJu/qwen3-asr-ctc-aiinfra` (a7_avg5/ subdirectory)
- FLEURS: `google/fleurs` (CC-BY-4.0), auto-fetched

Test-set availability — only FLEURS is fetched by the scripts themselves:

| dataset | how a third party gets it |
|---|---|
| FLEURS | automatic (`hf_hub_download`, cached under `HF_HOME`) |
| Buckeye | public on HF, but `runners/bench_buckeye.py` currently hardcodes a local snapshot path |
| LibriSpeech | public; `runners/qwen_librispeech_parity.py` currently hardcodes `/data/datasets/...` |
| AISHELL-1 | public (registration required); not wired into this repo |
| ASCEND | in-house corpus — **not publicly reproducible** |

None of these corpora are mirrored here; only derived reference text for the long-video sets
is tracked, because it is generated rather than published.

## Qwen3 CTC head revisions

There are 15 `qwen*` engine entries; three of them have published ONNX weights:

| round | engine | weights | fetch |
|---|---|---|---|
| v1 | `qwen` | HF `qwen-ctc/` | `download_models.py fetch` |
| aiinfra4 | `qwen_aiinfra4` | HF `qwen-ctc-aiinfra4/` | `download_models.py aiinfra4` |
| a7_avg5 (current) | `qwen_a7avg5` | HF `qwen-ctc-a7avg5/` | `download_models.py a7avg5` |
| r2 / ja / ja_sc / aiinfra / aiinfra2 / aiinfra3 / aiinfra5 | `qwen_r2` … | not published | export locally |

`models/qwen-ctc/` holds v1; unpublished heads go in their own `models/qwen-ctc-<round>/`
(all gitignored). The v1-vs-v2 comparison below is a **2026-08 historical experiment**:
v2 was a trade, not a general improvement — better English, worse Chinese.

| test set | metric | GLM | Qwen3 v1 | Qwen3 v2 |
|---|---|---|---|---|
| AISHELL-1 test | CER | **4.71%** | 5.31% | 5.53% |
| LibriSpeech test-clean | WER | **4.88%** | 6.93% | 6.53% |
| LibriSpeech test-other | WER | **9.99%** | 12.40% | 11.93% |
| ASCEND test | MER | **11.84%** | 14.53% | 14.47% |

The Chinese regression is real, not noise: 95% CI [+0.07, +0.34] pp over a
**sentence-level paired bootstrap** (2000 resamples), which does not contain 0.

> **What "sentence-level paired bootstrap" and "95% CI" mean**
>
> A test set is one fixed sample of sentences — what if it happens to favour one
> model? The bootstrap simulates "what if we drew a different set": resample the
> same number of sentences **with replacement**, score them, repeat 2000 times,
> and look at the spread of those 2000 results.
>
> **By sentence, not by character** — errors cluster inside an utterance (one bad
> sentence usually costs a dozen consecutive characters). Treating characters as
> independent samples overstates the effective sample size several times over and
> yields confidence intervals that are far too narrow.
>
> **Paired** — each resampled set of sentences is scored for **both models at
> once**, so "was this batch easy or hard" affects both sides equally and cancels
> in the difference. Measured on aishell1_test, pairing halves the CI width
> (0.263pp vs 0.529pp); for the same true difference of +0.208pp the paired
> interval excludes 0 while the unpaired one straddles it and can't call a
> direction.
>
> **95% CI** is the middle 95% of those 2000 differences. **Not containing 0**
> means the sign is stable no matter which sentences you draw; **straddling 0**
> means some parallel worlds favour A and others favour B — direction unknown,
> treat as noise.
>
> Implementation: `scripts/bootstrap_compare.py` in
> [glm-asr-ctc-train](https://github.com/JazerJu/glm-asr-ctc-train).

Weights: [qwen3-asr-ctc](https://huggingface.co/JazerJu/qwen3-asr-ctc) (v1, public) and
`qwen3-asr-ctc-r2` (v2, currently a private repository — external readers cannot open it).
Current head: [qwen3-asr-ctc-aiinfra/a7_avg5](https://huggingface.co/JazerJu/qwen3-asr-ctc-aiinfra/tree/main/a7_avg5).

Engineering conventions and invariants: [AGENTS.md](AGENTS.md).
