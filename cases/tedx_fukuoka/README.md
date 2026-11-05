# TEDx Fukuoka Case Study (aiinfra4)

Historical Japanese case study for [CapsWriter-Offline #457](https://github.com/HaujetZhao/CapsWriter-Offline/issues/457).
The aiinfra4 and Fun-ASR outputs here are not results for a7_avg5.

Audio is intentionally not redistributed. From the repository root:

```bash
python scripts/prepare_case_audio.py tedx_fukuoka --source /path/to/original.mp4
# Or (requires yt-dlp, ffmpeg, and access to YouTube):
python scripts/prepare_case_audio.py tedx_fukuoka --download
```

This creates mono 16 kHz PCM `audio16k.wav`, without overwriting an existing file.
The source URL is in `source.json`. Re-downloading may select a different media
encoding; preserve the original source for exact reproduction.

Saved JSON, SRT, reference subtitles, and comparison reports do not need audio
for inspection. `run_qwen.py` / `run_fun.py` are historical local runners with
machine-specific model paths. Adjust these paths and provision both runtimes
before re-running inference; preparing audio alone is not sufficient.
