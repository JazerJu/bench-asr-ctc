"""漏字位置分析：中文回归集 8 段 x 10 分钟，按 30 秒不重叠切窗，CTC 首遍逐窗转写，
与参考（v1 + LLM 二遍，逐字时间戳）做编辑距离对齐，记录每个被删掉的参考字在窗内的相对时间。

    python del_pos.py run <label> <head.onnx>     # 每个头单独一个进程
    python del_pos.py report v1 aiinfra4 aiinfra5 aiinfra6
"""
import csv, json, os, re, sys
from pathlib import Path
B = Path("/data/推理框架/asr-onnx/bench-asr-ctc")
CASE = os.environ.get("DELPOS_CASE", "zh_regression")
C = B / "cases" / CASE
OUT = B / "results"; W = 30.0
HAN = re.compile(r"[一-鿿]")

def align(ref, hyp):
    n, m = len(ref), len(hyp)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): D[i][0] = i
    for j in range(m + 1): D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i-1][j] + 1, D[i][j-1] + 1, D[i-1][j-1] + (ref[i-1] != hyp[j-1]))
    i, j, tag = n, m, [""] * n
    while i > 0:
        if j > 0 and D[i][j] == D[i-1][j-1] + (ref[i-1] != hyp[j-1]):
            tag[i-1] = "ok" if ref[i-1] == hyp[j-1] else "sub"; i -= 1; j -= 1
        elif D[i][j] == D[i-1][j] + 1:
            tag[i-1] = "del"; i -= 1
        else:
            j -= 1
    return tag

if sys.argv[1] == "run":
    label, head = sys.argv[2], Path(sys.argv[3])
    sys.path.insert(0, str(B / "scripts"))
    from en_regression import make_engine
    import soundfile as sf
    eng = make_engine(Path("/data/AI应用/流式转录/CapsWriter-Offline"), head, with_decoder=False)
    rows = []
    for c in csv.DictReader(open(C / "clips.tsv", encoding="utf-8"), delimiter="\t"):
        k = c["key"]
        wav, sr = sf.read(C / "audio" / f"{k}.wav", dtype="float32")
        words = json.load(open(C / "refs" / f"{k}.json", encoding="utf-8"))["words"]
        for w in range(int(len(wav) / sr // W)):
            a = w * W
            ref = [(ch, s - a) for ch, s, _ in words if a <= s < a + W and HAN.match(ch)]
            seg = wav[int(a * sr):int((a + W) * sr)]
            hyp = "".join(HAN.findall(eng.transcribe(seg).ctc_text))
            tag = align([ch for ch, _ in ref], hyp)
            rows.append({"clip": k, "win": w, "n_ref": len(ref), "n_hyp": len(hyp),
                         "del_t": [round(t, 2) for (ch, t), g in zip(ref, tag) if g == "del"],
                         "sub": tag.count("sub"), "ref": "".join(ch for ch, _ in ref),
                         "tag": "".join({"ok": ".", "sub": "s", "del": "d"}[g] for g in tag)})
        print(k, "done", flush=True)
    json.dump(rows, open(OUT / f"delpos_{CASE}_{label}.json", "w"), ensure_ascii=False)
    sys.exit()

labels = sys.argv[2:]
R = {l: json.load(open(OUT / f"delpos_{CASE}_{l}.json")) for l in labels}
nref = sum(r["n_ref"] for r in R[labels[0]])
print(f"共 {len(R[labels[0]])} 个 30 秒窗，参考汉字 {nref} 个")
print("\n漏字率（漏掉的参考字 / 参考字），按漏字在窗内的位置分档：")
print("%-10s" % "位置" + "".join("%10s" % l for l in labels))
for b in range(6):
    lo, hi = b * 5, b * 5 + 5
    cnt = {l: sum(1 for r in R[l] for t in r["del_t"] if lo <= t < hi) for l in labels}
    den = sum(1 for r in R[labels[0]] for _ in range(1))  # placeholder
    refn = None
    print("%-10s" % f"{lo}-{hi}s" + "".join("%10d" % cnt[l] for l in labels))
print("%-10s" % "合计" + "".join("%10d" % sum(len(r["del_t"]) for r in R[l]) for l in labels))
print("%-10s" % "漏字率" + "".join("%9.1f%%" % (sum(len(r["del_t"]) for r in R[l]) / nref * 100) for l in labels))
print("\n按窗口看漏字集中程度：漏字率 >50% 的窗口数 / 漏字率 <10% 的窗口数 / 这些重灾窗贡献的漏字占比")
for l in labels:
    rr = [r for r in R[l] if r["n_ref"] >= 20]
    heavy = [r for r in rr if len(r["del_t"]) / r["n_ref"] > 0.5]
    light = [r for r in rr if len(r["del_t"]) / r["n_ref"] < 0.1]
    tot = sum(len(r["del_t"]) for r in rr)
    print(f"  {l:10s} >50%: {len(heavy):3d} 窗   <10%: {len(light):3d} 窗   重灾窗占全部漏字 {sum(len(r['del_t']) for r in heavy)/max(1,tot)*100:.0f}%（共 {len(rr)} 窗）")

import collections
print("\n连续漏字的长度分布（一串连续 d 记一次）：")
print("%-10s" % "串长" + "".join("%10s" % l for l in labels))
runs = {}
for l in labels:
    c = collections.Counter()
    for r in R[l]:
        for m in re.finditer(r"d+", r.get("tag", "")): c[min(len(m.group()), 6)] += len(m.group())
    runs[l] = c
for k in range(1, 7):
    print("%-10s" % (f"{k}" if k < 6 else "≥6") + "".join("%10d" % runs[l][k] for l in labels))
print("\n（单位：漏掉的字数）\n\n各头漏得最多的字（相对 aiinfra4 增加最多的）：")
dc = {}
for l in labels:
    c = collections.Counter()
    for r in R[l]:
        for ch, g in zip(r.get("ref", ""), r.get("tag", "")):
            if g == "d": c[ch] += 1
    dc[l] = c
base = dc.get("aiinfra4", collections.Counter()); last = dc[labels[-1]]
inc = sorted(last, key=lambda ch: -(last[ch] - base[ch]))[:25]
print("  " + "  ".join(f"{ch}:{base[ch]}→{last[ch]}" for ch in inc))
