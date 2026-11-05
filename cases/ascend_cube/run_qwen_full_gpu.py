#!/usr/bin/env python3
import json, sys, time, soundfile as sf
from pathlib import Path

CASE = Path(__file__).resolve().parent
HOTWORDS = [x.strip() for x in (CASE / 'hot_ascend.txt').read_text().splitlines() if x.strip()]
wav, sr = sf.read(CASE / 'audio_full.wav', dtype='float32')
step = 30 * sr
n_chunks = (len(wav) + step - 1) // step

q_prev = json.load(open(CASE / 'out_qwen_10m30s.json'))
q_chunks = list(q_prev['chunks'])
q_aligned_all = list(q_prev['segments'])
start_chunk = len(q_chunks)

sys.path.insert(0, '/data/AI应用/流式转录/CapsWriter-Offline')
from util.qwen3_ctc_gguf import Qwen3ASREngine, ASREngineConfig, RecognitionStream

mdir = Path('/data/AI应用/流式转录/CapsWriter-Offline/models/Qwen3-ASR-CTC')
q_cfg = ASREngineConfig(
    encoder_onnx_path=str(mdir / 'Qwen3-ASR-Encoder.q4f16.onnx'),
    ctc_onnx_path=str(mdir / 'Qwen3-ASR-CTC.q4f16.onnx'),
    tokens_path=str(mdir / 'qwen-ctc-tokens.txt'),
    preprocessor_path=str(mdir),
    decoder_gguf_path=str(mdir / 'Qwen3-ASR-Decoder.q5_k_m.gguf'),
    use_gpu=True, llm_use_gpu=True,
    hotwords=HOTWORDS, similar_threshold=0.72, max_hotwords=10,
    n_ctx=1024, n_ubatch=256, n_outputs_max=1, kv_cache_type='q8_0',
    ort_gpu_mem_limit_mb=544, ort_shared_arena=True, ort_shared_stream=True,
    ort_mem_pattern=False, ctc_argmax_chunks=8,
)
q_eng = Qwen3ASREngine(q_cfg)
t0 = time.time()
for i in range(start_chunk, n_chunks):
    s = RecognitionStream(sample_rate=sr)
    s.accept_waveform(sr, wav[i*step:(i+1)*step])
    r = q_eng.decode_stream(s, temperature=0.0, timestamp_offset=i*30.0, verbose=False)
    q_chunks.append({
        'chunk': i, 'start': i*30.0, 'end': min((i+1)*30.0, len(wav)/sr),
        'ctc_text': r.ctc_text, 'hotwords': r.hotwords, 'llm_text': r.text,
    })
    q_aligned_all.extend(r.aligned)
    if (i + 1) % 10 == 0 or i == n_chunks - 1:
        print(f'[Qwen {i+1:02d}/{n_chunks}] elapsed={time.time()-t0:.1f}s hw={r.hotwords}')
q_elapsed = q_prev['elapsed'] + (time.time() - t0)
(CASE / 'out_qwen_full.json').write_text(json.dumps({
    'engine': 'qwen3_asr_ctc_aiinfra4', 'elapsed': q_elapsed,
    'hotwords_list': HOTWORDS, 'chunks': q_chunks, 'segments': q_aligned_all,
    'text': ''.join(c['llm_text'] for c in q_chunks),
    'ctc_text': ''.join(c['ctc_text'] for c in q_chunks),
}, ensure_ascii=False, indent=1))
print('QWEN FULL DONE in', time.time() - t0)
