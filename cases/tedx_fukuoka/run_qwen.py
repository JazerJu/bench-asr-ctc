#!/usr/bin/env python3
"""Qwen3-ASR CTC (aiinfra4) first-pass on TEDx audio -> json + srt."""
import json, sys, time
from pathlib import Path

GGUF_DIR = Path('/data/推理框架/asr-onnx/Qwen3-ASR-CTC-GGUF')
sys.path.insert(0, str(GGUF_DIR))
import os
os.environ['QWEN3_MODEL_DIR'] = str(GGUF_DIR / 'model-aiinfra4')
import export_config as C
from qwen3_asr_ctc import create_asr_engine
from qwen3_asr_ctc.srt import write_srt

CASE = Path(__file__).resolve().parent
enc = C.onnx(C.ENCODER, 'q4')
ctc = C.onnx(C.CTC, 'q4')
print('enc:', enc.exists(), 'ctc:', ctc.exists(), 'tokens:', C.TOKENS_TXT.exists(), 'preproc:', C.PREPROC_DIR.exists())
engine = create_asr_engine(
    encoder_onnx_path=str(enc), ctc_onnx_path=str(ctc),
    tokens_path=str(C.TOKENS_TXT), preprocessor_path=str(C.PREPROC_DIR),
    decoder_gguf_path=None, use_gpu=True, llm_use_gpu=False, hotwords=[],
)
t0 = time.time()
r = engine.transcribe(str(CASE / 'audio16k.wav'), language='Japanese')
print('elapsed', time.time() - t0, 'rtf', r.rtf, 'n_words', len(r.words))
out = {
    'engine': 'qwen_aiinfra4_ctc_q4_cuda',
    'text': r.text,
    'ctc_text': r.ctc_text,
    'words': [[w[0], w[1], w[2]] for w in r.words],
    'duration': r.duration,
}
(CASE / 'out_qwen_aiinfra4.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
write_srt(r.words, str(CASE / 'out_qwen_aiinfra4.srt'))
print('done')
