#!/usr/bin/env python3
"""Compare engine outputs vs manual Japanese subs (VidGo native_subtitles/1244.json)."""
import json, re, statistics, sys
from pathlib import Path
sys.path.insert(0, '/data/推理框架/asr-onnx/bench-asr-ctc')
from bench.metrics import normalize, edit_distance

CASE = Path(__file__).resolve().parent

ref = json.load(open(CASE / 'ref_subs.json'))['segments']
ref = [c for c in ref if c[0] > 5.0]          # skip translator-credit cue 0-7s
ref.sort(key=lambda c: c[0])

# windows: midpoint between adjacent cues
wins = []
for i, (s, e, t) in enumerate(ref):
    lo = (ref[i-1][1] + s) / 2 if i else 0.0
    hi = (e + ref[i+1][0]) / 2 if i < len(ref)-1 else e + 10.0
    wins.append((lo, s, e, hi, t))

def load_units(engine):
    if engine == 'qwen':
        d = json.load(open(CASE / 'out_qwen_aiinfra4.json'))
        return [(w[0], w[1], w[2]) for w in d['words']], d['text']
    if engine == 'fun':
        d = json.load(open(CASE / 'out_fun.json'))
        # segments: [char, onset] -> end = next onset
        segs = d['segments']
        units = [(c, t, segs[i+1][1] if i+1 < len(segs) else t + 0.1)
                 for i, (c, t) in enumerate(segs)]
        return units, d['text']

def windowed_eval(units):
    """Assign each unit to a ref window by onset; return per-cue pairs + orphan text."""
    per = [[] for _ in wins]
    orphan = []
    for text, s, e in units:
        hit = next((i for i, (lo, rs, re_, hi, t) in enumerate(wins) if lo <= s < hi), None)
        (per[hit] if hit is not None else orphan).append(text)
    err_sum = tot_sum = 0
    pairs = []
    for i, (lo, rs, re_, hi, tref) in enumerate(wins):
        thyp = ''.join(per[i])
        err, tot = edit_distance(normalize(tref, 'cer', 'ja_jp'), normalize(thyp, 'cer', 'ja_jp'), 'char')
        err_sum += err; tot_sum += tot
        pairs.append((rs, re_, tref, thyp, err, tot))
    return err_sum, tot_sum, pairs, ''.join(orphan)

def timing_eval(units):
    """Onset/offset error of first/last hyp unit within each ref window."""
    dstart, dend = [], []
    for lo, rs, re_, hi, t in wins:
        u = [u_ for u_ in units if lo <= u_[1] < hi]
        if not u:
            continue
        dstart.append(u[0][1] - rs)
        dend.append(u[-1][2] - re_)
    return dstart, dend

results = {}
for eng in ('qwen', 'fun'):
    units, text = load_units(eng)
    err, tot, pairs, orphan = windowed_eval(units)
    ds, de = timing_eval(units)
    results[eng] = dict(
        cer=err / tot * 100, err=err, total=tot,
        orphan_chars=len(normalize(orphan, 'cer', 'ja_jp')),
        onset_med=statistics.median(abs(x) for x in ds),
        onset_p90=sorted(abs(x) for x in ds)[int(len(ds)*0.9)],
        offset_med=statistics.median(abs(x) for x in de),
        offset_p90=sorted(abs(x) for x in de)[int(len(de)*0.9)],
        pairs=pairs,
    )

# Fun CTC first-pass text (no ts): global CER on ref-covered span only is not
# windowable; report raw global CER as a rough text-quality reference.
fun = json.load(open(CASE / 'out_fun.json'))
ref_all = ''.join(t for _, _, t in ref)
for name, hyp in [('fun_ctc_text', fun['ctc_text']), ('fun_final_text', fun['text'])]:
    err, tot = edit_distance(normalize(ref_all, 'cer', 'ja_jp'), normalize(hyp, 'cer', 'ja_jp'), 'char')
    results[name] = dict(cer_global=err / tot * 100, err=err, total=tot)

q = json.load(open(CASE / 'out_qwen_aiinfra4.json'))
err, tot = edit_distance(normalize(ref_all, 'cer', 'ja_jp'), normalize(q['text'], 'cer', 'ja_jp'), 'char')
results['qwen_ctc_text_global'] = dict(cer_global=err / tot * 100, err=err, total=tot)

print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'pairs'} for k, v in results.items()},
                 ensure_ascii=False, indent=1))

# worst cues per engine
for eng in ('qwen', 'fun'):
    print(f"\n=== {eng} worst cues (err>=5) ===")
    for rs, re_, tref, thyp, err, tot in sorted(results[eng]['pairs'], key=lambda p: -p[4])[:8]:
        if err < 5: continue
        print(f'[{rs:.1f}-{re_:.1f}] err={err}/{tot}')
        print(' REF:', tref)
        print(' HYP:', thyp)
