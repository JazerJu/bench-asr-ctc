#!/usr/bin/env python3
"""Run Qwen3-ASR-CTC (aiinfra4) and Fun-ASR-Nano on identical 30s chunks of
8b368e42ba7c47863d73122450cd0c4f.mp4 (00:00 - 10:30, 21 chunks) with identical
CTC PhonemeCorrector hotword list and parameters (similar_threshold=0.72, max_hotwords=10, temperature=0.0).
"""
import json, sys, time, onnxruntime, soundfile as sf
from pathlib import Path

CASE = Path(__file__).resolve().parent
HOTWORDS = [
    '昇腾', 'CANN', 'Ascend C', 'Ascend', 'Atlas A2', 'Atlas A3',
    'Cube', 'Matrix', 'Matmul', 'BatchMatmul', 'GroupedMatmul', 'FlashAttention',
    'Global Memory', 'AI Core', 'AIC', 'AIV',
    'L1 Buffer', 'L0A Buffer', 'L0B Buffer', 'L0C Buffer', 'BT Buffer', 'FP Buffer',
    'MTE1', 'MTE2', 'FixPipe', 'Scalar',
    'DataCopy', 'LoadData', 'Mmad', 'SetFlag', 'WaitFlag',
    'LocalMemAllocator', 'GetBlockIdx', 'Tiling', 'Double Buffer', 'msprof'
]
(CASE / 'hot_ascend.txt').write_text('\n'.join(HOTWORDS) + '\n', encoding='utf-8')

wav, sr = sf.read(CASE / 'audio_10m30s.wav', dtype='float32')
step = 30 * sr
n_chunks = (len(wav) + step - 1) // step
print(f'Loaded {len(wav)/sr:.1f}s audio -> {n_chunks} chunks of 30s, {len(HOTWORDS)} hotwords')

# 1. Run Qwen3-ASR-CTC (aiinfra4)
sys.path.insert(0, '/data/AI应用/流式转录/CapsWriter-Offline')
from util.qwen3_ctc_gguf import Qwen3ASREngine, ASREngineConfig, RecognitionStream

mdir = Path('/data/推理框架/asr-onnx/Qwen3-ASR-CTC-GGUF/model-aiinfra4')
pdir = Path('/data/推理框架/asr-onnx/Qwen3-ASR-CTC-GGUF/preprocessor')
q_cfg = ASREngineConfig(
    encoder_onnx_path=str(mdir / 'Qwen3-ASR-Encoder.q4.onnx'),
    ctc_onnx_path=str(mdir / 'Qwen3-ASR-CTC.q4.onnx'),
    tokens_path=str(mdir / 'tokens.txt'),
    preprocessor_path=str(pdir),
    decoder_gguf_path=str(mdir / 'Qwen3-ASR-Decoder.q5_k_m.gguf'),
    use_gpu=False, llm_use_gpu=False, n_threads=16,
    hotwords=HOTWORDS, similar_threshold=0.72, max_hotwords=10,
    n_ctx=1024, n_ubatch=256, n_outputs_max=1, kv_cache_type='q8_0',
)
q_eng = Qwen3ASREngine(q_cfg)
q_chunks, q_aligned_all = [], []
t0 = time.time()
for i in range(n_chunks):
    s = RecognitionStream(sample_rate=sr)
    s.accept_waveform(sr, wav[i*step:(i+1)*step])
    r = q_eng.decode_stream(s, temperature=0.0, timestamp_offset=i*30.0, verbose=False)
    q_chunks.append({
        'chunk': i, 'start': i*30.0, 'end': min((i+1)*30.0, len(wav)/sr),
        'ctc_text': r.ctc_text, 'hotwords': r.hotwords, 'llm_text': r.text,
    })
    q_aligned_all.extend(r.aligned)
    print(f'[Qwen {i+1:02d}/{n_chunks}] hw={r.hotwords} | llm={r.text[:45]}...')
q_elapsed = time.time() - t0
q_eng.cleanup()
del q_eng

(CASE / 'out_qwen_10m30s.json').write_text(json.dumps({
    'engine': 'qwen3_asr_ctc_aiinfra4', 'elapsed': q_elapsed,
    'hotwords_list': HOTWORDS, 'chunks': q_chunks, 'segments': q_aligned_all,
    'text': ''.join(c['llm_text'] for c in q_chunks),
    'ctc_text': ''.join(c['ctc_text'] for c in q_chunks),
}, ensure_ascii=False, indent=1))

# 2. Run Fun-ASR-Nano with identical PhonemeCorrector hotwords & threshold=0.72
sys.path.insert(0, '/data/推理框架/asr-onnx/Fun-ASR-GGUF')
orig_sess = onnxruntime.InferenceSession
def hybrid_sess(path, sess_options=None, providers=None, **kw):
    if 'Encoder' in str(path):
        providers = [('CUDAExecutionProvider', {'arena_extend_strategy': 'kSameAsRequested'}), 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']
    return orig_sess(path, sess_options=sess_options, providers=providers, **kw)
onnxruntime.InferenceSession = hybrid_sess

from fun_asr_gguf.inference import llama as f_llama
orig_init = f_llama.LlamaModel.__init__
def patched_init(self, path, n_gpu_layers=-1, use_gpu=1):
    orig_init(self, path, n_gpu_layers=0, use_gpu=False)
f_llama.LlamaModel.__init__ = patched_init

from fun_asr_gguf import FunASREngine, ASREngineConfig as FunConfig
from fun_asr_gguf.inference.schema import RecognitionStream as FunStream
from fun_asr_gguf.inference.display import DisplayReporter
from fun_asr_gguf.inference.srt_utils import generate_srt_file

f_mdir = Path('/data/推理框架/asr-onnx/Fun-ASR-GGUF/model')
f_int4 = Path('/data/AI应用/流式转录/CapsWriter-Offline/models/Fun-ASR-Nano/Fun-ASR-Nano-GGUF')
f_cfg = FunConfig(
    encoder_onnx_path=str(f_int4 / 'Fun-ASR-Nano-Encoder-Adaptor.int4.onnx'),
    ctc_onnx_path=str(f_mdir / 'Fun-ASR-Nano-CTC.int8.onnx'),
    decoder_gguf_path=str(f_mdir / 'Fun-ASR-Nano-Decoder.q5_k.gguf'),
    tokens_path=str(f_mdir / 'tokens.txt'),
    hotwords=HOTWORDS, similar_threshold=0.72, max_hotwords=10,
    enable_ctc=True, onnx_provider='CUDA', llm_use_gpu=False, n_threads=16, verbose=False,
)
f_eng = FunASREngine(f_cfg)
f_chunks, f_aligned_all = [], []
reporter = DisplayReporter(verbose=False)
t0 = time.time()
for i in range(n_chunks):
    s = FunStream()
    s.accept_waveform(sr, wav[i*step:(i+1)*step])
    r = f_eng.pipeline.decode_stream(s, language=None, context=None, verbose=False, reporter=reporter,
                                     timestamp_offset=i*30.0, temperature=0.0, top_p=1.0, top_k=1)
    ctc_t = ''.join(x.text for x in r.ctc_results) if r.ctc_results else ''
    f_chunks.append({
        'chunk': i, 'start': i*30.0, 'end': min((i+1)*30.0, len(wav)/sr),
        'ctc_text': ctc_t, 'hotwords': r.hotwords, 'llm_text': r.text,
    })
    f_aligned_all.extend(r.aligned)
    print(f'[Fun  {i+1:02d}/{n_chunks}] hw={r.hotwords} | llm={r.text[:45]}...')
f_elapsed = time.time() - t0
reporter.stop()
f_eng.cleanup()

(CASE / 'out_fun_10m30s.json').write_text(json.dumps({
    'engine': 'fun_asr_nano', 'elapsed': f_elapsed,
    'hotwords_list': HOTWORDS, 'chunks': f_chunks, 'segments': f_aligned_all,
    'text': ''.join(c['llm_text'] for c in f_chunks),
    'ctc_text': ''.join(c['ctc_text'] for c in f_chunks),
}, ensure_ascii=False, indent=1))

generate_srt_file(q_aligned_all, str(CASE / 'qwen_ascend_10m30s_nw.srt'))
generate_srt_file(f_aligned_all, str(CASE / 'fun_ascend_10m30s_nw.srt'))
print('ALL DONE')
