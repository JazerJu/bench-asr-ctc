#!/usr/bin/env python3
import json, sys, time, onnxruntime, soundfile as sf
from pathlib import Path

CASE = Path(__file__).resolve().parent
HOTWORDS = [x.strip() for x in (CASE / 'hot_ascend.txt').read_text().splitlines() if x.strip()]
wav, sr = sf.read(CASE / 'audio_full.wav', dtype='float32')
step = 30 * sr
n_chunks = (len(wav) + step - 1) // step

f_prev = json.load(open(CASE / 'out_fun_10m30s.json'))
f_chunks = list(f_prev['chunks'])
f_aligned_all = list(f_prev['segments'])
start_chunk = len(f_chunks)

sys.path.insert(0, '/data/推理框架/asr-onnx/Fun-ASR-GGUF')
orig_sess = onnxruntime.InferenceSession
def small_arena_sess(path, sess_options=None, providers=None, **kw):
    if providers and 'CUDAExecutionProvider' in providers:
        providers = [('CUDAExecutionProvider', {'arena_extend_strategy': 'kSameAsRequested'}), 'CPUExecutionProvider']
    return orig_sess(path, sess_options=sess_options, providers=providers, **kw)
onnxruntime.InferenceSession = small_arena_sess

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
    enable_ctc=True, onnx_provider='CUDA', llm_use_gpu=True, verbose=False,
)
f_eng = FunASREngine(f_cfg)
reporter = DisplayReporter(verbose=False)
t0 = time.time()
for i in range(start_chunk, n_chunks):
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
    if (i + 1) % 10 == 0 or i == n_chunks - 1:
        print(f'[Fun  {i+1:02d}/{n_chunks}] elapsed={time.time()-t0:.1f}s hw={r.hotwords}')
f_elapsed = f_prev['elapsed'] + (time.time() - t0)
reporter.stop()
(CASE / 'out_fun_full.json').write_text(json.dumps({
    'engine': 'fun_asr_nano', 'elapsed': f_elapsed,
    'hotwords_list': HOTWORDS, 'chunks': f_chunks, 'segments': f_aligned_all,
    'text': ''.join(c['llm_text'] for c in f_chunks),
    'ctc_text': ''.join(c['ctc_text'] for c in f_chunks),
}, ensure_ascii=False, indent=1))

q_full = json.load(open(CASE / 'out_qwen_full.json'))
generate_srt_file(q_full['segments'], str(CASE / 'qwen_ascend_full_nw.srt'))
generate_srt_file(f_aligned_all, str(CASE / 'fun_ascend_full_nw.srt'))
print('FUN FULL DONE in', time.time() - t0)
