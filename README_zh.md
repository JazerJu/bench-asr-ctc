# bench-asr-ctc

[English](README.md) · [中文](README_zh.md)

跨模型的 CTC **首遍**对比评测:**GLM-ASR(我们训练的 CTC 头) vs Fun-ASR-Nano vs Qwen3-ASR(我们训练的 CTC 头)** —— 三个引擎,同一套接口。

同一批音频上测三件事:

1. **转写质量** —— FLEURS 官方 test split(11 个语种),WER / CER
2. **热词 DP 匹配** —— CapsWriter 那套 pypinyin 音素 + 模糊编辑距离(CTC 原文永不修改,命中的热词只作为 LLM prompt 的提示上报)
3. **可复现性** —— 每个引擎都在同一个接口后面(`encode` → `decode_text`),int4 对 int4 公平比

## a7_avg5 更新

HF 的 ONNX / PyTorch 权重、model card 与校验清单均已上传；使用下方命令单独下载 a7，原有 `fetch` 不受影响。

新增 self-conditioned Qwen3 CTC 头，引擎名 `qwen_a7avg5`，默认 q4（与其他轮一致）；HQQ 用 `--precision hqqf16`。
针对 [CapsWriter #457](https://github.com/HaujetZhao/CapsWriter-Offline/issues/457)
反馈的英文音频串出中文问题做了数据清洗、续训和末段 5 个 checkpoint 平均。
旧的 `qwen` 和 `qwen_aiinfra4` 保留，不覆盖旧权重。

```bash
python scripts/download_models.py a7avg5
python bench.py --counts 20 --engines qwen_a7avg5 --langs en_us,cmn_hans_cn
python scripts/summarize_a7.py  # 只读已保存的结果，不需要音频、模型或 GPU
```

中文 17 段回归集上，a7_avg5 HQQ CER 为 **14.96%**（a6_avg5 HQQ 为 16.20%），
英文 6 段为 **5.98%**（6.68%）。参考是 v1+LLM 伪标签，数字是逐视频宏平均，
不等于人工真值上的独立测试成绩。TED 29 段的 CTC 汉字总数降至 6，但并非零；
其中多篇与 TED-LIUM 训练数据重叠，只作回归检查。

[版本、精度、原始结果与音频准备说明](docs/a7_avg5.md)。
长视频音频不入库，重新推理前需准备原始媒体；只汇总保存的结果不需要音频。
下表仍是各自原版本的历史成绩，**没有当作 a7_avg5 的新成绩**。

## 历史结果(三个引擎都是 int4,FLEURS **官方 test 全量** 7,876 句,全部跑 CUDA EP)

> 这里是 **v1 时期** 的数字(2026-08/09):Qwen3-CTC 是 56,916 步的 v1 头,GLM 是 `final134k`。
> 它们不是 a7_avg5 的成绩 —— 当前头见 [docs/a7_avg5.md](docs/a7_avg5.md)。

| 语种 | 指标 | n | GLM-CTC | Fun-ASR-Nano | Qwen3-CTC | 胜者 |
|---|---|---|---|---|---|---|
| en_us | WER | 647 | 18.3% | **18.1%** | 18.3% | Fun(差 0.2pp,平手) |
| cmn_hans_cn | CER | 945 | 10.7% | **9.3%** | 10.9% | Fun |
| ko_kr | CER | 382 | 39.5% | 46.0% | **20.3%** | **Qwen(好一倍)** |
| ja_jp | CER | 650 | 31.0% | **17.0%** | 30.9% | Fun |
| yue_hant_hk | CER | 819 | 42.7% | 43.2% | **30.1%** | **Qwen** |
| de_de | WER | 862 | **35.7%** | 75.5% | 41.6% | GLM |
| fr_fr | WER | 676 | **39.8%** | 76.6% | 46.6% | GLM |
| es_419 | WER | 908 | **29.7%** | 63.0% | 35.6% | GLM |
| it_it | WER | 865 | **40.8%** | 86.2% | 50.9% | GLM |
| nl_nl | WER | 364 | **46.7%** | 95.8% | 50.1% | GLM |
| pl_pl | WER | 758 | **69.3%** | 98.8% | 77.1% | GLM |

**GLM 6 胜 / Fun 3 胜 / Qwen 2 胜。** 原始数据 `results/fleurs_full_3way.json`。

Buckeye 口语英文(全量 2,477 段 / 20 位说话人 / 145,690 词,跳过 1 段空):**Qwen 18.1% < GLM 22.4% < Fun 26.9%**(`results/buckeye_full.json`)。自发口语仍是 Qwen 占优,GLM 明显好于 Fun。早先那个 40 段 s01 切片(Qwen 23.3 / GLM 36.2 / Fun 41.4)是碰上了难说话人,不代表整个语料。朗读英文(FLEURS en)三家仍是平手。

**中文与中英夹杂**(FLEURS 子切分,`runners/bench_mixed.py`,三方):普通话纯 CER Fun 8.4 < GLM 9.7 ≈ Qwen 9.8;普通话 mixed-CER Fun 17.0 < GLM 19.2 ≈ Qwen 19.1 —— wiki 那种朗读文本上 Fun 保持小幅领先。粤语(含嵌入英文):Qwen 纯 29.6 / 混 37.7,领先另外两家 12pp 以上。口语技术词召回(fq 视频,37 个窗口):Fun 54% > Qwen 32% > GLM 27%,而且**漏的词是互补的** —— Qwen 能抓到 GLM 漏的 bbr/gfw/cpu,GLM 能抓到 Qwen 漏的 ssl(`results/spoken_term_recall_3way.json`)。

Qwen3-CTC 的几点说明:编码器参数只有 GLM 的一半(317.5M vs 635.0M),帧率 13fps(GLM 是 50fps)—— 每秒音频的计算量大约是 GLM 的 1/8。它在 ko/yue 上的优势来自 Qwen3-ASR 底座的多语种预训练;GLM 与 Qwen 在中英上的差距则要归到训练配方(同一批数据,512×56,916 对 256×134,140 次更新,对照实验待做)。

关于 provider 的注意:Fun 的 int4 ONNX **不是 provider 不变的** —— ORT CPU 与 CUDA EP 的转写在约 5% 的样本上不同(最多差 1.9pp CER,CPU 更弱,见 `runners/parity_check.py`)。上表所有数字都是三个引擎统一跑 CUDA EP。a7_avg5 的 RTN q4/q4f16 与 HQQ hqqf16 同样是
MatMulNBits 图,DirectML 要用 fp16 图(`--precision fp16`)。

复现:`python bench.py --counts 200 --engines glm fun qwen`(每引擎约 2,200 句,采样是确定性的等距抽取 —— 同样的 `--counts` 永远评同一批句子)。

## 安装

```bash
pip install -r requirements.txt          # onnxruntime-gpu 可选

# Linux / NVIDIA CUDA(int4,默认):
python scripts/download_models.py fetch

# Windows 核显 DirectML(fp16,不含 MatMulNBits):
#   pip install onnxruntime-directml
#   python scripts/download_models.py fetch-fp16
#   python bench.py --counts 20 --provider dml --precision fp16 --engines glm,qwen
#    = glm-ctc/  取自 JazerJu/glm-asr-ctc-bench(编码器 q4、投影器、CTC final134k q4、tokens)
#    + fun-asr/  取自 JazerJu/glm-asr-ctc-bench/fun-asr(int4 三件套;FunAudioLLM 的模型,
#                int4 转换沿用 CapsWriter-Offline 的打包)
#    + qwen-ctc/ 取自 JazerJu/glm-asr-ctc-bench/qwen-ctc(编码器 q4 + CTC q4 +
#                tokens.txt + preprocessor_config.json)
#
# 一条命令下完三个引擎,外加 aiinfra4 的头。a7_avg5 是单独命令:
#   python scripts/download_models.py a7avg5
# fp16 变体加 FETCH_FP16=1,GLM 的 fp32 加 FETCH_FP32=1。
# models/ 下的一切都由这一步产生,仓库里不跟踪任何权重。
```

## 运行

```bash
# 冒烟:2 个语种 × 20 句(约 30 秒)
python bench.py --counts 20 --langs en_us,cmn_hans_cn

# issue 表格规模:11 语种 × 200 句(GPU 上约 20 分钟)
python bench.py --counts 200

# 官方 test 全量(7,876 句)
python bench.py --counts full

# 单条音频推理,任意后端(只跑 CTC 首遍,任意 ffmpeg 格式,≤30 秒)
python infer.py clip.wav --engine glm
python infer.py input.mp3 --engine fun --cpu

# 在自带样例上跑热词 DP 匹配(4 段 30 秒,期望命中已编码在内)
python runners/bench_hotword.py --engines glm fun

# fp32 一致性检查(仅 GLM)
python bench.py --counts 20 --engines glm --fp32 --langs en_us
```

每份结果 JSON 都记了 `meta` 块(引擎版本、量化档、ORT 版本、execution provider、时间戳)。FLEURS 的 parquet 分片由 `huggingface_hub` 自动拉取并缓存在 `HF_HOME`,不用手动下数据集。

## 目录

```
bench/
  models.py        # 引擎注册表:GLMEngine / FunEngine / QwenEngine(含各续训轮)
  metrics.py       # WER/CER + 按语种的归一化
  hotword/         # PhonemeCorrector DP 匹配(源自 CapsWriter)
runners/
  bench_fleurs.py  # 多语种转写评测
  bench_hotword.py # 热词命中/漏报/误报检查
cases/             # 4 段 30 秒片段(cases.json + 热词)与长视频回归集:
                   #   en_regression/ zh_regression/ zh_long20/ ted_collection/
                   #   tedx_fukuoka/ ascend_cube/(音频不入库,见各自 README)
docs/
  a7_avg5.md       # 当前头:结果、精度、音频可得性
tests/
  test_release.py  # 引擎路径、下载布局、音频准备
models/
  glm-ctc/         # encoder.q4.onnx(+.data)、projector、ctc-final134k.q4.onnx、tokens
  fun-asr/         # Fun-ASR-Nano int4 三件套(来自 JazerJu/glm-asr-ctc-bench/fun-asr)
  qwen-ctc/        # 共用编码器 q4、v1 头、tokens、preprocessor
  qwen-ctc-aiinfra4/  # aiinfra4 头(q4/q4f16)—— download_models.py aiinfra4
  qwen-ctc-a7avg5/    # a7_avg5 头(q4/q4f16/hqqf16)—— download_models.py a7avg5
scripts/
  download_models.py  # fetch / aiinfra4 / a7avg5 / fetch-fp16 / fun / publish
  en_regression.py    # 长音频回归(需 CapsWriter 运行时)
  ted_collection_check.py / summarize_a7.py / prepare_case_audio.py / del_pos.py
```

## Qwen 引擎

`QwenEngine`(`qwen`)是 v1:冻结的官方 Qwen3-ASR 编码器(int4,13fps,固定 30 秒桶的 ONNX)
+ 我们 56,916 步的 CTC 头([`JazerJu/qwen3-asr-ctc`](https://huggingface.co/JazerJu/qwen3-asr-ctc))。
ONNX 链路对训练侧 PyTorch 的一致性:LibriSpeech test-clean WER 6.92% 对 6.93%
(`runners/qwen_librispeech_parity.py`)。

后续各轮共用同一份冻结编码器和词表,只换头:`qwen_aiinfra4`(2026-09)与
`qwen_a7avg5`(当前版,修 #457 的英文音频串中文)。结果与口径见
[docs/a7_avg5.md](docs/a7_avg5.md)。导出细节见 `../Qwen3-ASR-CTC-GGUF/`。

## 模型来源

- GLM 编码器/投影器:从 `zai-org/GLM-ASR-Nano-2512`(MIT)导出,int4 量化由我们做
- GLM CTC 头:我们训练的(134k 步,warmup + 5 层 transformer block,BPE 59264),发布在 `JazerJu/glm-asr-ctc-bench`
- Fun-ASR-Nano:HaujetZhao/CapsWriter-Offline 的模型套件(int4 三件套镜像在 JazerJu/glm-asr-ctc-bench/fun-asr)
- Qwen3 编码器:官方 `Qwen/Qwen3-ASR-1.7B` 的 audio tower(冻结),int4 由我们做
- Qwen3 CTC 头均为我们训练:v1 在 `JazerJu/qwen3-asr-ctc`;aiinfra4 与 a7_avg5 在
  `JazerJu/qwen3-asr-ctc-aiinfra`(a7_avg5/ 子目录)
- FLEURS:`google/fleurs`(CC-BY-4.0),自动拉取

测试集可得性 —— 脚本自身只会拉 FLEURS:

| 数据集 | 第三方怎么拿到 |
|---|---|
| FLEURS | 自动(`hf_hub_download`,缓存在 `HF_HOME`) |
| Buckeye | HF 上公开,但 `runners/bench_buckeye.py` 目前写死了本地快照路径 |
| LibriSpeech | 公开;`runners/qwen_librispeech_parity.py` 目前写死 `/data/datasets/...` |
| AISHELL-1 | 公开(需注册);未接进本仓库 |
| ASCEND | 自建语料 —— **外部无法复现** |

这些语料都没有镜像到本仓库;仓库里只跟踪长视频回归集的参考文本,因为那是生成的、不是公开发布的。

## Qwen3 CTC 权重索引

`qwen*` 一共 15 个引擎条目,其中三轮发布了 ONNX 权重:

| 轮次 | 引擎名 | 权重位置 | 下载 |
|---|---|---|---|
| v1 | `qwen` | HF `qwen-ctc/` | `download_models.py fetch` |
| aiinfra4 | `qwen_aiinfra4` | HF `qwen-ctc-aiinfra4/` | `download_models.py aiinfra4` |
| a7_avg5(当前) | `qwen_a7avg5` | HF `qwen-ctc-a7avg5/` | `download_models.py a7avg5` |
| r2 / ja / ja_sc / aiinfra / aiinfra2 / aiinfra3 / aiinfra5 | `qwen_r2` 等 | 未发布 | 自行导出 |

`models/qwen-ctc/` 放 v1;未发布的头各放 `models/qwen-ctc-<轮次>/`(均已 gitignore)。

权重:[qwen3-asr-ctc](https://huggingface.co/JazerJu/qwen3-asr-ctc)(v1,公开)与
`qwen3-asr-ctc-r2`(v2,目前是 private 仓库,外部读者打不开)。
当前头:[qwen3-asr-ctc-aiinfra/a7_avg5](https://huggingface.co/JazerJu/qwen3-asr-ctc-aiinfra/tree/main/a7_avg5)。

## 2026-08 的 v1 与 v2 对照(历史)

v2 是一笔交易而非普遍提升 —— 英文更好、中文更差:

| 测试集 | 指标 | GLM | Qwen3 v1 | Qwen3 v2 |
|---|---|---|---|---|
| AISHELL-1 test | CER | **4.71%** | 5.31% | 5.53% |
| LibriSpeech test-clean | WER | **4.88%** | 6.93% | 6.53% |
| LibriSpeech test-other | WER | **9.99%** | 12.40% | 11.93% |
| ASCEND test | MER | **11.84%** | 14.53% | 14.47% |

工程约定与不变量见 [AGENTS.md](AGENTS.md)。
