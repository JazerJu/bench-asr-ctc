# 英文回归集

aiinfra4 在部分英文音频上 CTC 首遍丢词，NW 对齐缺锚点，SRT 字幕成段错位。
FLEURS 和 LibriSpeech 都是朗读短句，抓不到这个问题，所以单独固定这 6 条。

| key | 内容 | 时长 |
| --- | --- | ---: |
| ted | TED: Teach girls bravery, not perfection（Reshma Saujani） | 12:44 |
| redcross | British Red Cross × Databricks 宣传片 | 2:25 |
| cosmos | NVIDIA Cosmos 3 产品介绍 | 3:13 |
| planning | 类脑 agent 规划架构讲解 | 6:03 |
| moe | What is Mixture of Experts? | 7:57 |
| slm | LLMs vs SLMs: A developer's guide | 11:23 |

视频在 VidGo 里（`clips.tsv` 有哈希和来源链接），`audio/` 由脚本按需从 VidGo 抽出，不进库。

`ted` 这条在 TED-LIUM 3 的训练集里（`ReshmaSaujani_2016`），训练 manifest
`tedlium3_train.jsonl` 已把它剔除。往这里加新的 TED 时，同样要先从训练集里剔掉。

## 参考与指标

`refs/<key>.json` 是 v1 头 + LLM 二遍的完整输出（最终文本 + 逐词时间戳）。
LLM 二遍的最终文本和 CTC 头无关（CTC 文本只用来抽热词、不进 prompt），
所以各个头的最终文本逐字相同、字幕条数相同，可以按序号逐条比起始时间。

| 指标 | 含义 |
| --- | --- |
| CTC 字数比 | CTC 首遍字符数 / 参考字符数（只数 a-z0-9），丢词时明显下降 |
| CER | CTC 首遍对参考的字符错误率 |
| 汉字 | CTC 首遍里的汉字个数 |
| SRT >1s / >2s | 字幕条起始时间与 v1 基线相差超过 1 / 2 秒的条数 |

```bash
python scripts/en_regression.py --head <Qwen3-ASR-CTC.q4f16.onnx> --label <名字>
```

## 基线（2026-10-22，.92 上 CapsWriter 引擎）

| 头 | ted SRT >1s / >2s | redcross 字数比 / CER | cosmos / planning / moe / slm CER |
| --- | --- | --- | --- |
| v1（q4） | 0 / 0 | 94.8% / 7.0% | 21.6 / 9.0 / 5.2 / 7.5 |
| aiinfra1（q4） | 5 / 2 | 70.9% / 29.8% | 18.0 / 4.0 / 3.6 / 7.8 |
| aiinfra2（q4） | 3 / 1 | 69.6% / 30.7% | 18.0 / 4.2 / 3.5 / 8.6 |
| aiinfra3（q4f16） | 2 / 1 | 90.8% / 10.7% | 16.3 / 4.0 / 3.8 / 6.6 |
| aiinfra4（q4f16） | 9 / 4 | 49.8% / 51.0% | 14.4 / 4.2 / 3.4 / 7.1 |

ted 上最大 12.3 秒那一条，所有 aiinfra 头都一样（645–657 s 重复的 "Thank you"），
是对齐器在重复短语上锚错位置，和 CTC 头无关，不计入验收。

aiinfra1/2 没有导出 q4f16，只有 q4；同一个头 q4 和 q4f16 的结果会差几个点，比较时注意精度。

## aiinfra5 的验收线（按 aiinfra3 的水平，q4f16）

- ted：SRT >1s 不多于 3 条，>2s 不多于 1 条
- redcross：CTC 字数比 ≥ 85%，CER ≤ 15%
- 其余 4 条：CER 不比 aiinfra3 差 2 个百分点以上
