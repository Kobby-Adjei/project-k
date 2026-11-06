#!/usr/bin/env python3
"""Build the docs-sounds clips (Nov 6, 2026).

Same recipe as pilot2/swap/build_swap.py, copied so this check
is self-contained: each natural recording is cut to exactly
10 s, loudness matched to about RMS -20 dB with a measured
gain loop plus limiter guard, then muxed onto (a) the swap
still picture and (b) a plain gray 0x808080 frame. Every
finished MP4 is decoded back and measured.
Outputs beside this script: docs_<sound>_still.mp4,
docs_<sound>_gray.mp4, measurements.json.
"""
import hashlib, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, 'work')
SRC = os.path.join(HERE, 'sources')
STILL = os.path.join(HERE, '..', 'pilot2', 'swap', 'work', 'still.png')
TARGET_RMS = -20.0
TOL = 0.7
SR = '48000'

SOUNDS = {  # name: (source file, window start s)
    'birds': ('birds_robin.mp3', 34.0),
    'applause': ('applause.mp3', 1.0),
    'fireworks': ('fireworks.mp3', 2.0),
}


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print('FAILED:', ' '.join(cmd), file=sys.stderr)
        print(r.stderr[-2000:], file=sys.stderr)
        sys.exit(1)
    return r


def measure_rms(path):
    cmd = ['ffmpeg', '-hide_banner', '-i', path, '-map', '0:a:0',
           '-af', 'astats=metadata=1', '-f', 'null', '-']
    r = subprocess.run(cmd, capture_output=True, text=True)
    vals = re.findall(r'RMS level dB:\s*(-?[\d.]+|-inf)', r.stderr)
    if not vals:
        return None
    v = vals[-1]
    return float('-inf') if v == '-inf' else float(v)


def extract_audio(src, dst, start, dur=10.0):
    run(['ffmpeg', '-y', '-v', 'error', '-ss', str(start), '-t', str(dur),
         '-i', src, '-vn', '-ac', '2', '-ar', SR, dst])


def match_loudness(src_wav, dst_wav):
    gain = 0.0
    rms = measure_rms(src_wav)
    if rms is None or rms == float('-inf'):
        run(['ffmpeg', '-y', '-v', 'error', '-i', src_wav, dst_wav])
        return rms, 0.0
    for _ in range(6):
        gain += TARGET_RMS - rms
        af = f'volume={gain:.2f}dB,alimiter=limit=0.84:level=0'
        run(['ffmpeg', '-y', '-v', 'error', '-i', src_wav,
             '-af', af, '-ac', '2', '-ar', SR, dst_wav])
        rms = measure_rms(dst_wav)
        if rms is not None and abs(rms - TARGET_RMS) <= TOL:
            break
    return rms, gain


def mux(frame, wav, out_mp4):
    run(['ffmpeg', '-y', '-v', 'error', '-loop', '1', '-framerate', '25',
         '-i', frame, '-i', wav, '-t', '10',
         '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-r', '25',
         '-c:a', 'aac', '-ar', SR, '-shortest', out_mp4])


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    if not os.path.exists(STILL):
        print('MISSING swap still:', STILL, file=sys.stderr)
        sys.exit(1)
    gray = os.path.join(WORK, 'gray.png')
    run(['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi',
         '-i', 'color=c=0x808080:s=854x480:d=1', '-frames:v', '1', gray])
    measurements = {}
    for name, (src_name, start) in SOUNDS.items():
        raw_wav = os.path.join(WORK, f'a_{name}_raw.wav')
        matched_wav = os.path.join(WORK, f'a_{name}_matched.wav')
        extract_audio(os.path.join(SRC, src_name), raw_wav, start)
        rms, gain = match_loudness(raw_wav, matched_wav)
        entry = {'source': src_name, 'window_start_s': start,
                 'matched_wav_rms_db': None if rms is None else round(rms, 2),
                 'gain_applied_db': round(gain, 2), 'versions': {}}
        for version, frame in (('still', STILL), ('gray', gray)):
            out_mp4 = os.path.join(HERE, f'docs_{name}_{version}.mp4')
            mux(frame, matched_wav, out_mp4)
            pr = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
                                 'format=duration', '-show_entries',
                                 'stream=codec_type,codec_name', '-of', 'json',
                                 out_mp4], capture_output=True, text=True)
            info = json.loads(pr.stdout)
            dur = float(info['format']['duration'])
            streams = [(s['codec_type'], s['codec_name'])
                       for s in info['streams']]
            dec_rms = measure_rms(out_mp4)
            entry['versions'][version] = {
                'file': os.path.basename(out_mp4),
                'duration_s': round(dur, 3),
                'streams': streams,
                'audio_rms_db_decoded': None if dec_rms is None else round(dec_rms, 2),
                'sha256': sha256(out_mp4),
            }
            print(f'{name} {version}: dur={dur:.2f} rms={dec_rms}', flush=True)
        measurements[name] = entry
    with open(os.path.join(HERE, 'measurements.json'), 'w') as f:
        json.dump(measurements, f, indent=2)
    print('DONE')


if __name__ == '__main__':
    main()
