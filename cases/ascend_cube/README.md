# Ascend Cube Case Study (aiinfra4)

Historical case study for [CapsWriter-Offline #457](https://github.com/HaujetZhao/CapsWriter-Offline/issues/457),
not an a7_avg5 re-run. The original video is listed in `source.json`.

Audio is intentionally not redistributed. From the repository root:

```bash
python scripts/prepare_case_audio.py ascend_cube --source /path/to/original.mp4
# Or, if you can access the public source (requires yt-dlp and ffmpeg):
python scripts/prepare_case_audio.py ascend_cube --download
```

This creates mono 16 kHz PCM `audio_full.wav` and its first 630 seconds
(`audio_10m30s.wav`). Existing audio is not overwritten. Re-downloading may select
a different media encoding; preserve your original source for exact reproduction.

`out_*_full.json`, `*_full_nw.srt`, and `hot_ascend.txt` are saved outputs and the
shared hotword list. They can be inspected without downloading any media.
`run_*.py` are historical local runners, not portable entry points: they also
require the referenced CapsWriter/Fun-ASR runtime, decoder weights, and path
adjustments. `run_both_full.py` resumes from the saved 630-second outputs; it is
not a fresh full-video runner. Preparing audio alone does not provision models.
