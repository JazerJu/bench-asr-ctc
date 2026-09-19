# -*- coding: utf-8 -*-
"""FLEURS 汇总（aiinfra4 收尾轮）：各语种错误率、微/宏平均、含 U+FFFD 的句数、日语叠字率。"""
import json, sys
sys.path.insert(0, "/data/推理框架/asr-onnx/bench-asr-ctc/scripts")
from dup_rate import dup_rate

LANGS = ["en_us", "cmn_hans_cn", "ko_kr", "ja_jp", "yue_hant_hk", "de_de", "fr_fr", "es_419", "it_it", "nl_nl", "pl_pl"]
E = ["qwen_ja_sc", "qwen_aiinfra2", "qwen_aiinfra3", "qwen_aiinfra4"]
NAME = {"qwen_ja_sc": "基座", "qwen_aiinfra2": "aiinfra2", "qwen_aiinfra3": "aiinfra3", "qwen_aiinfra4": "aiinfra4"}


def averages(res, eng):
    e = t = 0; rates = []
    for l in LANGS:
        a, b = res[l]["%s_err/total" % eng]; e += a; t += b; rates.append(100 * a / b)
    return 100 * e / t, sum(rates) / len(rates)


abl = json.load(open("/data/推理框架/asr-onnx/bench-asr-ctc/results/fleurs_ablation.json"))["results"]
print("口径自检：ablation 里 qwen_ja_sc 微平均 %.2f 宏平均 %.2f（笔记 30.39 / 34.15）" % averages(abl, "qwen_ja_sc"))

j = json.load(open(sys.argv[1]))
res, meta = j["results"], j["meta"]
hyps = json.load(open(sys.argv[2], encoding="utf-8"))
print("onnxruntime %s  providers %s" % (meta.get("onnxruntime"), set(meta.get("providers", {}).values())))
print("\n| 语种 | 指标 | n | 基座 | aiinfra2 | aiinfra3 | aiinfra4 | 4−基座 | 4−3 |")
print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
worse_b, worse_3 = [], []
for l in LANGS:
    r = res[l]
    v = {e: 100 * r["%s_err/total" % e][0] / r["%s_err/total" % e][1] for e in E}
    d1, d2 = v[E[3]] - v[E[0]], v[E[3]] - v[E[2]]
    if d1 > 0: worse_b.append("%s %+.2f" % (l, d1))
    if d2 > 0: worse_3.append("%s %+.2f" % (l, d2))
    print("| %s | %s | %d | %.2f | %.2f | %.2f | %.2f | %+.2f | %+.2f |" % (l, r["metric"], r["n"], *[v[e] for e in E], d1, d2))
avg = {e: averages(res, e) for e in E}
print("| **微平均** | | | %s | %+.2f | %+.2f |" % (" | ".join("%.2f" % avg[e][0] for e in E), avg[E[3]][0] - avg[E[0]][0], avg[E[3]][0] - avg[E[2]][0]))
print("| **宏平均** | | | %s | %+.2f | %+.2f |" % (" | ".join("%.2f" % avg[e][1] for e in E), avg[E[3]][1] - avg[E[0]][1], avg[E[3]][1] - avg[E[2]][1]))
print("\naiinfra4 相对基座变差：%s" % (", ".join(worse_b) or "无"))
print("aiinfra4 相对 aiinfra3 变差：%s" % (", ".join(worse_3) or "无"))

print("\n| 语种 | 句数 | 含 � 的句（基座/2/3/4） | � 字符数（同序） |")
print("|---|---:|---|---|")
for l in LANGS:
    items = hyps[l]
    segs = [sum("�" in x[e] for x in items) for e in E]
    chars = [sum(x[e].count("�") for x in items) for e in E]
    if any(chars):
        print("| %s | %d | %s | %s |" % (l, len(items), " / ".join(map(str, segs)), " / ".join(map(str, chars))))

items = hyps["ja_jp"]
clean = lambda t: t.replace("�", "")
rd, rt = dup_rate([x["ref"] for x in items])
out = []
for e in E:
    d1, t1 = dup_rate([x[e] for x in items]); d2, t2 = dup_rate([clean(x[e]) for x in items])
    out.append("%s 原样 %.2f%% / 删�后 %.2f%%" % (NAME[e], 100 * d1 / t1, 100 * d2 / t2))
print("\n日语叠字率（n=%d，参考 %.2f%%）：%s" % (len(items), 100 * rd / rt, "；".join(out)))
