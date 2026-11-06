# a7_avg5 Release And Regression Results

Publication status: code/results and both HF payloads are published.
ONNX revision: `659fb65a8bcb30bd419b23bae97f060abcb03500`;
PyTorch revision: `2a286f7c3ebbd99130cb64e71d94d33e1859f36c`.
The new file hashes and retention of the historical files have been verified.

This is a new self-conditioned Qwen3 CTC head, not an overwrite of aiinfra4.
Only a7_avg5 weights are added in this release; a5/a6 results are retained as
historical comparisons, without publishing their weights.

- ONNX: [JazerJu/glm-asr-ctc-bench/qwen-ctc-a7avg5](https://huggingface.co/JazerJu/glm-asr-ctc-bench/tree/main/qwen-ctc-a7avg5)
- PyTorch: [JazerJu/qwen3-asr-ctc-aiinfra/a7_avg5](https://huggingface.co/JazerJu/qwen3-asr-ctc-aiinfra/tree/main/a7_avg5) (the aiinfra-series repo; its root stays aiinfra4)
- Shared encoder, byte vocabulary, feature extractor: the existing
  [qwen-ctc/](https://huggingface.co/JazerJu/glm-asr-ctc-bench/tree/main/qwen-ctc).
- Issue context: [CapsWriter-Offline #457](https://github.com/HaujetZhao/CapsWriter-Offline/issues/457).

## What Changed

Some aiinfra training segments paired English speech with Chinese translation
subtitles. The a6 data cleaning removed 2,445 suspect segments. a7 continued
from a6, with Chinese long clips sampled x2 and short clips x3, for approximately
80k additional updates. This release averages steps **412000, 414000, 416000,
418000, 420000**, rather than selecting the lowest validation-loss checkpoint.
The frozen encoder and compact vocabulary are unchanged. The self-conditioned
head has 85,448,596 parameters; it is not the original 48.3M v1 head.

The historical export's `config.json` records the first averaged step (412000)
and a placeholder `val_loss=0.0`; this is not a measured zero validation loss.
The release manifest records the complete averaging list.

## Regression Table

These are CTC first-pass results on CUDA EP. The CapsWriter regression runner
kept its existing shared q4f16 encoder; only the head was exchanged.

| Head | zh17 CER | zh17 character ratio | en6 CER | TED CTC Han characters | Parts with Han |
|---|---:|---:|---:|---:|---:|
| a6_avg5 fp16 | 15.96% | 89.1% | 6.39% | 6 | 3/29 |
| a6_avg5 RTN q4f16 | 17.73% | 87.0% | 6.82% | 7 | 4/29 |
| a6_avg5 HQQ hqqf16 | 16.20% | 89.2% | 6.68% | 11 | 6/29 |
| a7_avg5 fp16 | 14.35% | 91.1% | 5.59% | 5 | 3/29 |
| a7_avg5 RTN q4f16 | 15.07% | 90.4% | 6.29% | 5 | 4/29 |
| a7_avg5 HQQ hqqf16 | 14.96% | 90.7% | 5.98% | 6 | 4/29 |

Recompute this table from saved outputs, without audio/models/GPU:

```bash
python scripts/summarize_a7.py
```

Raw files: `results/zh_long20_{a6avg5,a7avg5}_*.json`,
`results/en_regression_{a6avg5,a7avg5}_*.json`,
`results/ted_collection_{a6avg5,a7avg5}_*.json`.

### Measurement Boundaries

- zh17 excludes `v669`, `v623`, `v736` from the historical 20-clip set because
  those three clips are predominantly English/German.
- CER and character ratio are **unweighted means over clips**. Punctuation and
  whitespace are removed; Latin letters are lowercased. These references are
  saved v1+LLM outputs, **not manually verified ground truth**.
- TED Han counts refer to **CTC output**, not the decoder's final subtitles.
  Residual Han characters remain: this is a large reduction, not zero leakage.
- Many TED talks overlap TED-LIUM training. These are **regression results**,
  not held-out English quality scores. Zero hits in a handful of sampled text
  windows do not establish that a talk is unseen.
- CTC CER does not directly measure NW timestamp quality. Timestamp comparisons
  in the raw results are relative to v1, not independently annotated boundaries.
- The older FLEURS, Japanese TEDx, and Ascend Cube tables are historical results
  of their named versions, not new a7_avg5 results.

## Download And Run

```bash
python scripts/download_models.py a7avg5
python infer.py cases/zh_news_30s.wav --engine qwen_a7avg5 --cpu
python bench.py --counts 20 --langs en_us,cmn_hans_cn --engines qwen_a7avg5

# Explicitly choose a CTC precision; encoder stays the shared q4 graph:
python bench.py --counts 20 --engines qwen_a7avg5 --precision q4f16
FETCH_FP16=1 python scripts/download_models.py a7avg5
python bench.py --counts 20 --engines qwen_a7avg5 --precision fp16
```

The a7 engine defaults to **q4**, matching other rounds; select **hqqf16** explicitly with `--precision hqqf16`.
In this bench registry the shared encoder is q4, whereas the saved CapsWriter
regression table uses q4f16. Do not expect bit-identical numbers across them.
For DirectML, run `fetch-fp16` for the shared encoder and download the new head
with `FETCH_FP16=1 python scripts/download_models.py a7avg5`; use
`--provider dml --precision fp16`. That path is prepared but has **not been validated on
Windows/DirectML hardware**. MatMulNBits q4/HQQ graphs are not DirectML graphs.

## Audio Availability

No new long-video audio is redistributed. This does not affect existing bundled
short cases or automatic FLEURS downloads.

The [Japanese TEDx](../cases/tedx_fukuoka/README.md) and
[Ascend Cube](../cases/ascend_cube/README.md) cases have public source metadata
and `scripts/prepare_case_audio.py` for a local source or an explicitly requested
yt-dlp download. Their historical inference scripts still need model/runtime
paths adjusted.

[zh_long20](../cases/zh_long20/README.md) and
[TED collection](../cases/ted_collection/README.md) use original media or prepared
WAVs. In particular, zh_long20 has local hashes but incomplete public source
URLs: a fresh clone without those media cannot reproduce its inference run.
Saved-result summaries remain runnable without any audio or local services.
