"""WER/CER metrics + language-specific text normalization (shared across runners)."""

from __future__ import annotations

import re


def normalize(text: str, metric: str, lang: str) -> str:
    t = text.lower()
    if metric == "cer":
        if lang in ("cmn_hans_cn", "yue_hant_hk"):
            return re.sub(r"[^\u4e00-\u9fff]", "", t)
        if lang == "ko_kr":
            return re.sub(r"[^가-힣ㄱ-ㅎㅏ-ㅣ]", "", t)
        if lang == "ja_jp":
            return re.sub(r"[^\u3040-\u30ff\u4e00-\u9fff]", "", t)
        return t
    t = re.sub(r"[^a-zàâäéèêëîïôöùûüçñáíóúãõâêôòìú' ]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def edit_distance(ref: str, hyp: str, unit: str):
    r = list(ref) if unit == "char" else ref.split()
    h = list(hyp) if unit == "char" else hyp.split()
    n, m = len(r), len(h)
    if n == 0:
        return m, 0
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (0 if r[i - 1] == h[j - 1] else 1))
            prev = cur
    return dp[m], n


def error_rate(ref: str, hyp: str, metric: str, lang: str):
    unit = "char" if metric == "cer" else "word"
    ref_n, hyp_n = normalize(ref, metric, lang), normalize(hyp, metric, lang)
    err, total = edit_distance(ref_n, hyp_n, unit)
    return err, total
