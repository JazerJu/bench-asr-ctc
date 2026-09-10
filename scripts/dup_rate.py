#!/usr/bin/env python3
"""叠字率：相邻两字相同的位置占全文的比例。

CTC 假设各帧条件独立，模型没有机制保证相邻帧预测自洽。典型表现是吐
`X ␣ X`（中间夹一个 blank），CTC 的合并规则「先去重复、再去 blank」把它
还原成 `XX` —— 一个凭空多出来的叠字。参考文本本身也有叠字（日语里
「様々」「時々」这类），所以要拿 ref 当底线，看模型比 ref 多出多少。

用法：
    python scripts/dup_rate.py results/fleurs_ja_sc_hyps.json --lang ja_jp
"""
import argparse, json, sys
from pathlib import Path


def dup_rate(texts):
    """返回 (叠字数, 总字数)。空白不计入，避免分词差异干扰。"""
    dup = tot = 0
    for t in texts:
        s = "".join(t.split())
        tot += len(s)
        dup += sum(1 for i in range(1, len(s)) if s[i] == s[i - 1])
    return dup, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hyps", nargs="+")
    ap.add_argument("--lang", default="ja_jp")
    args = ap.parse_args()

    rows = {}
    for path in args.hyps:
        d = json.load(open(path, encoding="utf-8"))
        if args.lang not in d:
            print(f"{path}: 没有 {args.lang}", file=sys.stderr)
            continue
        items = d[args.lang]
        keys = [k for k in items[0] if k != "ref"]
        rows.setdefault("ref", dup_rate([x["ref"] for x in items]))
        for k in keys:
            rows[k] = dup_rate([x[k] for x in items if k in x])

    print(f"{args.lang}   n={len(items)} 句")
    print("%-22s %8s %10s %9s" % ("来源", "叠字", "总字", "叠字率"))
    base = rows["ref"][0] / rows["ref"][1]
    for k, (dup, tot) in rows.items():
        r = dup / tot
        extra = "" if k == "ref" else f"   (比 ref 多 {(r - base) * 100:+.2f}pp)"
        print("%-22s %8d %10d %8.2f%%%s" % (k, dup, tot, r * 100, extra))


if __name__ == "__main__":
    main()
