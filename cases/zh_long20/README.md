# zh_long20 Regression Set

`clips.tsv` defines 20 ten-minute clips, starting at 60 seconds in each original
video. Despite the historical set name, three clips are predominantly English
or German: `v669`, `v623`, `v736`. The published Chinese subset uses the other
**17 clips**; do not average all 20 and label it the Chinese result.

`refs/*.json` contains v1+LLM pseudo-reference text and timestamps, not manual
ground truth. CER includes lowercase Latin letters/digits and CJK characters;
punctuation and spaces are removed. Summary CER and character ratio are
unweighted means over the 17 clips.

Audio is not bundled:

- Existing WAVs go in `audio/<key>.wav` (mono, 16 kHz).
- With the original VidGo media, `--video-dir` extracts the manifest's exact
  start/duration automatically.
- This historical manifest contains VidGo media hashes; its source URL fields
  are empty. A fresh clone without those media cannot fully reproduce inference.
  There is no automatic public download path for this set.

```bash
python scripts/en_regression.py --case zh_long20 \
  --head /path/to/Qwen3-ASR-CTC.hqqf16.onnx --label a7avg5_hqqf16 \
  --capswriter /path/to/CapsWriter-Offline --video-dir /path/to/saved_video

# Recompute the released comparison from saved outputs, with no media or GPU:
python scripts/summarize_a7.py
```

The runner requires the CapsWriter Qwen CTC integration, decoder weights, and
shared encoder. Inspect `make_engine()` for the exact filenames. Replacing the
runtime/encoder changes the experimental condition; record those changes.
