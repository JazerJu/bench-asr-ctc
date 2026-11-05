#!/usr/bin/env python3
"""Fun-ASR-Nano full pipeline on TEDx audio (temperature=0.0) -> json + srt."""
import json, sys, time, onnxruntime
from pathlib import Path

sys.path.insert(0, '/data/推理框架/asr-onnx/Fun-ASR-GGUF')

orig_sess = onnxruntime.InferenceSession
def hybrid_sess(path, sess_options=None, providers=None, **kw):
    if 'Encoder' in str(path):
        providers = [('CUDAExecutionProvider', {'arena_extend_strategy': 'kSameAsRequested'}), 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']
    return orig_sess(path, sess_options=sess_options, providers=providers, **kw)
onnxruntime.InferenceSession = hybrid_sess

from fun_asr_gguf.inference import llama
orig_init = llama.LlamaModel.__init__
def patched_init(self, path, n_gpu_layers=-1, use_gpu=1):
    orig_init(self, path, n_gpu_layers=0, use_gpu=False)
llama.LlamaModel.__init__ = patched_init

from fun_asr_gguf import FunASREngine, ASREngineConfig

CASE = Path(__file__).resolve().parent
model_dir = Path('/data/推理框架/asr-onnx/Fun-ASR-GGUF/model')
int4_dir = Path('/data/AI应用/流式转录/CapsWriter-Offline/models/Fun-ASR-Nano/Fun-ASR-Nano-GGUF')

config = ASREngineConfig(
    encoder_onnx_path=str(int4_dir / 'Fun-ASR-Nano-Encoder-Adaptor.int4.onnx'),
    ctc_onnx_path=str(model_dir / 'Fun-ASR-Nano-CTC.int8.onnx'),
    decoder_gguf_path=str(model_dir / 'Fun-ASR-Nano-Decoder.q5_k.gguf'),
    tokens_path=str(model_dir / 'tokens.txt'),
    hotwords=[],
    enable_ctc=True,
    onnx_provider='CUDA',
    llm_use_gpu=False,
    n_threads=16,
    verbose=False,
)
engine = FunASREngine(config)
t0 = time.time()
result = engine.transcribe(
    str(CASE / 'audio16k.wav'),
    language='日文',
    verbose=False,
    segment_size=30.0,
    overlap=2.0,
    temperature=0.0,
)
print('elapsed', time.time() - t0)
out = {
    'engine': 'fun_asr_nano_temp0_full_pipeline',
    'text': result.text,
    'ctc_text': getattr(result, 'ctc_text', None),
    'segments': result.segments,
    'hotwords': getattr(result, 'hotwords', []),
}
(CASE / 'out_fun.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
engine.cleanup()
print('done')
