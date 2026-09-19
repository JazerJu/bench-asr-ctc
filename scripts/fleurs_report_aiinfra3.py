# -*- coding: utf-8 -*-
"""FLEURS 全量结果汇总：各语种错误率、相对 ja_sc / aiinfra2 的变化、微平均 / 宏平均、含 U+FFFD 的句数、日语叠字率。
    python fleurs_report.py results/fleurs_aiinfra3.json results/fleurs_aiinfra3_hyps.json
口径自检：先用 results/fleurs_ablation.json 复算 qwen_ja_sc 的微平均 / 宏平均，应为 30.39 / 34.15（笔记里的数）。
"""
import json, sys
sys.path.insert(0, "/data/推理框架/asr-onnx/bench-asr-ctc/scripts")
from dup_rate import dup_rate

LANGS = ["en_us", "cmn_hans_cn", "ko_kr", "ja_jp", "yue_hant_hk", "de_de", "fr_fr", "es_419", "it_it", "nl_nl", "pl_pl"]


def averages(res, eng):
    e = t = 0; rates = []
    for l in LANGS:
        a, b = res[l]["%s_err/total" % eng]; e += a; t += b; rates.append(100 * a / b)
    return 100 * e / t, sum(rates) / len(rates)


abl = json.load(open("/data/推理框架/asr-onnx/bench-asr-ctc/results/fleurs_ablation.json"))["results"]
mi, ma = averages(abl, "qwen_ja_sc")
print("口径自检：ablation 里 qwen_ja_sc 微平均 %.2f 宏平均 %.2f（笔记 30.39 / 34.15）" % (mi, ma))

res = json.load(open(sys.argv[1]))["results"]; meta = json.load(open(sys.argv[1]))["meta"]
hyps = json.load(open(sys.argv[2], encoding="utf-8"))
engs = ["qwen_ja_sc", "qwen_aiinfra", "qwen_aiinfra2", "qwen_aiinfra3"]
print("onnxruntime %s providers %s" % (meta.get("onnxruntime"), meta.get("providers")))
print("\n| 语种 | 指标 | n | ja_sc | aiinfra | aiinfra2 | aiinfra3 | aiinfra3−ja_sc | aiinfra3−aiinfra2 |")
print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
worse_sc, worse_a2 = [], []
for l in LANGS:
    r = res[l]; v = {e: 100 * r["%s_err/total" % e][0] / r["%s_err/total" % e][1] for e in engs}
    d1, d2 = v["qwen_aiinfra3"] - v["qwen_ja_sc"], v["qwen_aiinfra3"] - v["qwen_aiinfra2"]
    if d1 > 0: worse_sc.append("%s %+.2f" % (l, d1))
    if d2 > 0: worse_a2.append("%s %+.2f" % (l, d2))
    print("| %s | %s | %d | %.2f | %.2f | %.2f | %.2f | %+.2f | %+.2f |" % (l, r["metric"], r["n"], v["qwen_ja_sc"], v["qwen_aiinfra"], v["qwen_aiinfra2"], v["qwen_aiinfra3"], d1, d2))
avg = {e: averages(res, e) for e in engs}
print("| **微平均** | | | %s | %+.2f | %+.2f |" % (" | ".join("%.2f" % avg[e][0] for e in engs), avg["qwen_aiinfra3"][0] - avg["qwen_ja_sc"][0], avg["qwen_aiinfra3"][0] - avg["qwen_aiinfra2"][0]))
print("| **宏平均** | | | %s | %+.2f | %+.2f |" % (" | ".join("%.2f" % avg[e][1] for e in engs), avg["qwen_aiinfra3"][1] - avg["qwen_ja_sc"][1], avg["qwen_aiinfra3"][1] - avg["qwen_aiinfra2"][1]))
print("\naiinfra3 相对 ja_sc 变差：%s" % (", ".join(worse_sc) or "无"))
print("aiinfra3 相对 aiinfra2 变差：%s" % (", ".join(worse_a2) or "无"))

print("\n| 语种 | 句数 | 含 � 的句（ja_sc / aiinfra / aiinfra2 / aiinfra3） | � 字符数（同序） |")
print("|---|---:|---|---|")
for l in LANGS:
    items = hyps[l]
    segs = [sum("�" in x[e] for x in items) for e in engs]
    chars = [sum(x[e].count("�") for x in items) for e in engs]
    if any(chars):
        print("| %s | %d | %s | %s |" % (l, len(items), " / ".join(map(str, segs)), " / ".join(map(str, chars))))

items = hyps["ja_jp"]
rd, rt = dup_rate([x["ref"] for x in items])
print("\n日语叠字率（n=%d）：参考 %.2f%%；%s" % (len(items), 100 * rd / rt, "；".join(
    "%s %.2f%%（比参考多 %+.2fpp）" % (e, 100 * dup_rate([x[e] for x in items])[0] / dup_rate([x[e] for x in items])[1],
                                    100 * dup_rate([x[e] for x in items])[0] / dup_rate([x[e] for x in items])[1] - 100 * rd / rt) for e in engs)))
