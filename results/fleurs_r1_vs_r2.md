# Qwen3-ASR-CTC r2 vs r1 (FLEURS full, int4 encoder + r2 CTC q4-blocks, CUDA EP)

r2 CTC: ffn_hidden 2048, 58.2M params, q4 152.7 MiB (ctc_lo kept fp32). Encoder reused r1 q4.

| lang | r1 | r2 | Fun | GLM |
|---|---|---|---|---|
| en_us WER | 18.3 | **18.2** | 18.1 | 18.3 |
| cmn CER | **10.9** | 11.3 | 9.3 | 10.7 |
| ko CER | 20.3 | **19.2** | 46.0 | 39.5 |
| ja CER | **30.9** | 31.6 | 17.0 | 31.0 |
| yue CER | 30.1 | **27.4** | 43.2 | 42.7 |
| de WER | 41.6 | **37.3** | 75.5 | 35.7 |
| fr WER | 46.6 | **42.0** | 76.6 | 39.8 |
| es WER | 35.6 | **29.9** | 63.0 | 29.7 |
| it WER | 50.9 | **43.7** | 86.2 | 40.8 |
| nl WER | 50.1 | **47.2** | 95.8 | 46.7 |
| pl WER | 77.1 | **73.4** | 98.8 | 69.3 |

CTC latency 5070 Ti (warm median): r2 30/50 ms (5s/30s) ≈ r1 32/51.

Isolated CTC int4 vs fp32 (same encoder q4): mean char-diff 0.12%.
