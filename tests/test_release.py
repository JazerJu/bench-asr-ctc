import contextlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench import models
from bench.models import QwenA7Avg5Engine, QwenAiinfra4Engine


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReleaseTests(unittest.TestCase):
    def _paths_for(self, cls, **env):
        recorded = []

        def fake(path, use_gpu, intra_threads=6):
            recorded.append(Path(path).name)
            return Mock()

        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "qwen-ctc").mkdir()
            (Path(td) / "qwen-ctc" / "tokens.txt").write_text("", encoding="utf-8")
            fake_transformers = types.ModuleType("transformers")
            fake_transformers.WhisperFeatureExtractor = Mock(
                from_pretrained=Mock(return_value=Mock())
            )
            with patch.object(models, "_session", side_effect=fake), \
                 patch.dict(os.environ, env, clear=True), \
                 patch.dict(sys.modules, {"transformers": fake_transformers}), \
                 patch.object(models, "MODELS_DIR", Path(td)):
                cls(use_gpu=False)
        return recorded

    def test_a7_paths_use_existing_precision_contract(self):
        self.assertEqual(self._paths_for(models.QwenA7Avg5Engine),
                         ["Qwen3-ASR-Encoder.q4.onnx", "Qwen3-ASR-CTC.q4.onnx"])
        self.assertEqual(self._paths_for(models.QwenA7Avg5Engine, BENCH_PRECISION="hqqf16"),
                         ["Qwen3-ASR-Encoder.q4.onnx", "Qwen3-ASR-CTC.hqqf16.onnx"])
        self.assertEqual(self._paths_for(models.QwenA7Avg5Engine, BENCH_PRECISION="fp16"),
                         ["Qwen3-ASR-Encoder.q4.onnx", "Qwen3-ASR-CTC.fp16.onnx"])

    def test_a7_is_a_plain_subclass(self):
        self.assertIs(models.QwenA7Avg5Engine.__mro__[1], models.QwenAiinfra2Engine)
        self.assertNotIn("model_paths", models.QwenA7Avg5Engine.__dict__)
        self.assertEqual(models.QwenA7Avg5Engine.CTC_SUBDIR, "qwen-ctc-a7avg5")

    def test_historical_engine_default_is_unchanged(self):
        self.assertEqual(self._paths_for(models.QwenAiinfra4Engine),
                         ["Qwen3-ASR-Encoder.q4.onnx", "Qwen3-ASR-CTC.q4.onnx"])

    def test_download_fetchers_mirror_existing_pattern(self):
        mod = load_script("download_models")
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {}, clear=True):
            source = Path(td) / "fake-download"
            source.write_bytes(b"weights")
            with patch.object(mod, "ROOT", Path(td)), \
                 patch.object(mod, "A7AVG5_DIR", Path(td) / "models/qwen-ctc-a7avg5"), \
                 patch.object(mod, "hf_hub_download", return_value=str(source)) as download, \
                 contextlib.redirect_stdout(io.StringIO()):
                mod.fetch_a7avg5()
                self.assertEqual(download.call_count, 4)
                self.assertTrue((Path(td) / "models/qwen-ctc-a7avg5/Qwen3-ASR-CTC.hqqf16.onnx").is_file())
                self.assertTrue(hasattr(mod, "fetch_aiinfra4"))
                self.assertEqual(mod.AIINFRA4_FILES[0], "qwen-ctc-aiinfra4/Qwen3-ASR-CTC.q4.onnx")

    def test_audio_preparation_does_not_overwrite(self):
        mod = load_script("prepare_case_audio")
        with tempfile.TemporaryDirectory() as td:
            case = Path(td)
            (case / "source.json").write_text(json.dumps({"outputs": {"out.wav": {"duration": 630}}}))
            source = case / "source.mp4"
            source.touch()
            with patch.object(mod.subprocess, "run") as run:
                mod.prepare(case, source)
                command = run.call_args.args[0]
                self.assertIn("-n", command)
                self.assertEqual(command[command.index("-t") + 1], "630")
                (case / "out.wav").write_bytes(b"original")
                run.reset_mock()
                mod.prepare(case, source)
                run.assert_not_called()
                self.assertEqual((case / "out.wav").read_bytes(), b"original")

    def test_missing_media_has_actionable_error(self):
        mod = load_script("en_regression")
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(FileNotFoundError, "Audio is not bundled"):
                mod.ensure_wav(Path(td), {"key": "clip", "vidgo_hash": "missing"}, Path(td))

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg required")
    def test_real_audio_preparation_format_and_cut(self):
        mod = load_script("prepare_case_audio")
        with tempfile.TemporaryDirectory() as td:
            case = Path(td)
            source = case / "source.wav"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                            "sine=frequency=440:duration=1", "-ar", "44100",
                            "-ac", "2", str(source)], check=True)
            (case / "source.json").write_text(json.dumps({
                "outputs": {"out.wav": {"start": 0.1, "duration": 0.25}},
            }))
            mod.prepare(case, source)
            probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of",
                                    "json", str(case / "out.wav")],
                                   check=True, capture_output=True, text=True)
            stream = json.loads(probe.stdout)["streams"][0]
            self.assertEqual(int(stream["sample_rate"]), 16000)
            self.assertEqual(stream["channels"], 1)
            self.assertAlmostEqual(float(stream["duration"]), 0.25, places=2)

    def test_ted_manifest_needs_no_live_service(self):
        mod = load_script("ted_collection_check")
        with patch("urllib.request.urlopen", side_effect=AssertionError("network not allowed")):
            self.assertEqual(len(mod.list_parts()), 29)

    def test_fleurs_runner_records_a7_head_and_precision(self):
        spec = importlib.util.spec_from_file_location("fleurs_runner", ROOT / "runners/bench_fleurs.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        audio = io.BytesIO()
        with wave.open(audio, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\0\0" * 160)
        table = Mock(num_rows=1)
        table.to_pylist.return_value = [{
            "audio": {"bytes": audio.getvalue()}, "transcription": "hello",
        }]
        engine = QwenA7Avg5Engine.__new__(QwenA7Avg5Engine)
        engine.enc_sess = Mock()
        engine.ctc_sess = Mock()
        engine.enc_sess.get_providers.return_value = ["CPUExecutionProvider"]
        engine.ctc_sess.get_providers.return_value = ["CPUExecutionProvider"]
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {}, clear=True):
            out = Path(td) / "smoke.json"
            args = types.SimpleNamespace(engines="qwen_a7avg5", gpu=False, fp32=False,
                                         langs="en_us", per_lang="1", out=str(out),
                                         dump_hyps=None)
            with patch.object(runner, "get_engine", return_value=engine), \
                 patch.object(engine, "encode", return_value=None), \
                 patch.object(engine, "decode_text", return_value="hello"), \
                 patch("huggingface_hub.hf_hub_download", return_value="fixture.parquet"), \
                 patch("pyarrow.parquet.read_table", return_value=table):
                runner.run(args)
            meta = json.loads(out.read_text())["meta"]
            self.assertIn("a7_avg5", meta["engines"]["qwen_a7avg5"]["ctc"])
            self.assertTrue(meta["engines"]["qwen_a7avg5"]["quantized"])


if __name__ == "__main__":
    unittest.main()
