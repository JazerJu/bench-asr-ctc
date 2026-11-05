# TED Collection Regression (29 Parts)

Source: [BV1pBp46rEQh](https://www.bilibili.com/video/BV1pBp46rEQh/).
`clips.tsv` fixes the part numbers, titles, media hashes, and original part URLs;
the runner no longer requires a live VidGo service just to discover the list.

Audio is not bundled. Supply mono 16 kHz `audio/p01.wav` ... `audio/p29.wav`,
or pass `--video-dir` containing the original hash-named MP4s. Public source
URLs are provided for obtaining the media yourself; access/availability is not
guaranteed.

```bash
python scripts/ted_collection_check.py \
  --head a7avg5_hqqf16=/path/to/Qwen3-ASR-CTC.hqqf16.onnx \
  --capswriter /path/to/CapsWriter-Offline --video-dir /path/to/saved_video

# Saved-output comparison: no audio, models, VidGo, or GPU needed.
python scripts/summarize_a7.py
```

The historical `--report ... --tedlium /path/to/texts.tsv` contamination check
also works without audio or VidGo. `--vidgo-api http://host:port` explicitly
opts into live discovery instead of the manifest.

The headline is the number of **CTC first-pass Han characters on English
audio**, not final decoder text, CER, or a timestamp-accuracy measurement.
The total is not zero in a7_avg5. Much of this collection overlaps TED-LIUM
used during training: it is a regression set, **not a held-out English test**.
Eight sampled text windows with zero training hits do not prove a talk was
excluded from training.
