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

from bench.models import QwenA7Avg5Engine, QwenAiinfra4Engine


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReleaseTests(unittest.TestCase):
    def test_a7_paths(self):
        engine = QwenA7Avg5Engine.__new__(QwenA7Avg5Engine)
        with patch.dict(os.environ, {}, clear=True):
            enc, ctc = engine.model_paths()
            self.assertEqual(enc.name, "Qwen3-ASR-Encoder.q4.onnx")
            self.assertEqual(ctc.name, "Qwen3-ASR-CTC.hqqf16.onnx")
            for tag in ("fp16", "q4", "q4f16", "hqqf16"):
                os.environ["BENCH_PRECISION"] = tag
                self.assertEqual(engine.model_paths()[1].name, f"Qwen3-ASR-CTC.{tag}.onnx")
                self.assertEqual(engine.model_paths()[0], enc)

    def test_dml_requires_unquantized_graph(self):
        engine = QwenA7Avg5Engine.__new__(QwenA7Avg5Engine)
        with patch.dict(os.environ, {"BENCH_ORT_PROVIDER": "dml"}, clear=True):
            with self.assertRaisesRegex(ValueError, "DirectML"):
                engine.model_paths()
            os.environ["BENCH_PRECISION"] = "fp16"
            self.assertEqual(engine.model_paths()[0].name, "Qwen3-ASR-Encoder.fp16.onnx")
        with patch.dict(os.environ, {"BENCH_PRECISION": "typo"}, clear=True):
            with self.assertRaises(ValueError):
                engine.model_paths()

    def test_historical_engine_default_is_unchanged(self):
        engine = QwenAiinfra4Engine.__new__(QwenAiinfra4Engine)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(engine.model_paths()[1].name, "Qwen3-ASR-CTC.q4.onnx")

    def test_download_head_layout(self):
        mod = load_script("download_models")
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {}, clear=True):
            source = Path(td) / "fake-download"
            source.write_bytes(b"weights")
            with patch.object(mod, "ROOT", Path(td)), patch.object(
                mod, "hf_hub_download", return_value=str(source)
            ) as download, contextlib.redirect_stdout(io.StringIO()):
                mod.fetch_head("qwen-ctc-a7avg5")
                self.assertEqual(download.call_count, 4)
                self.assertTrue((Path(td) / "models/qwen-ctc-a7avg5/Qwen3-ASR-CTC.hqqf16.onnx").is_file())
                download.reset_mock()
                mod.fetch_head("qwen-ctc-a7avg5", fp16_only=True)
                self.assertEqual(download.call_args.args[1], "qwen-ctc-a7avg5/Qwen3-ASR-CTC.fp16.onnx")

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
            self.assertEqual(meta["engines"]["qwen_a7avg5"]["precision"], "hqqf16")
            self.assertIn("a7_avg5", meta["engines"]["qwen_a7avg5"]["ctc"])
            self.assertEqual(meta["ctc_providers"]["qwen_a7avg5"], ["CPUExecutionProvider"])


if __name__ == "__main__":
    unittest.main()
