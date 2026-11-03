#!/usr/bin/env python3
"""Build project_k_minicpm_series2_colab.ipynb.

Series Two diagnostic notebook for MiniCPM-o 2.6. It reuses the proven
sealed-venv MiniCPM notebook pattern: same private environment, same
pinned hub revision, same loading ladder, and full failure capture.
The battery is different. Every ask carries a feed proof printed from
inside the session before the answer is taken, and the unit builder
hard-stops on a missing, empty, short, or wrongly silent track instead
of silently padding it.

The repaired JSON question is not retyped. It is pulled from
pilot2/run_eggs.py at build time and asserted at 444 characters.
Plain language throughout, no em dashes.
"""
import copy
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT2 = os.path.join(HERE, "..", "pilot2")
BASE_NB = os.path.join(HERE, "project_k_minicpm_o_colab.ipynb")


def extract_prompt(path, varname):
    src = open(path).read()
    m = re.search(varname + r" = \((.*?)\)\n", src, re.S)
    if not m:
        raise SystemExit(f"could not find {varname} in {path}")
    ns = {}
    exec(varname + " = (" + m.group(1) + ")", ns)
    return ns[varname]


PROMPT_JSON = extract_prompt(os.path.join(PILOT2, "run_eggs.py"), "PROMPT")
assert len(PROMPT_JSON) == 444, f"repaired prompt drifted: {len(PROMPT_JSON)}"


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip().splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip().splitlines(True)}


with open(BASE_NB) as fh:
    base_nb = json.load(fh)

# The MiniCPM notebook cells are: 0 title, 1 Step 1 markdown, 2 Step 1
# code. Step 1 is reused byte for byte because the environment, stack,
# flash-attn wheel, and proof are deliberately identical.
step1_md = copy.deepcopy(base_nb["cells"][1])
step1_code = copy.deepcopy(base_nb["cells"][2])

cells = []
cells.append(md(r"""
# Project K, Series Two: MiniCPM-o 2.6 diagnostic run

**Why this notebook exists.** Series One showed that MiniCPM-o 2.6
reported water and missed most other sounds. The council then asked a
harder question: which stage loses the sound? The feed audit (Series
Two test 1) proved the old runner had a silent failure path: a missing
track could be padded with zeros and leave no trace. This notebook is
built so that cannot happen again.

**What this notebook does.** It runs three diagnostic tests on the
same MiniCPM-o 2.6, through the same door as the Series One run:

1. **Test 2, the 2x2.** Four sounds (speech, true silence, water,
   beeps), each fed as sound alone and as sound with the swap still
   picture, crossed with an open question and direct questions.
2. **Test 3, yes/no scoring.** The full seven-file swap set, in three
   versions each: sound on with the still picture, muted with the
   still picture, and sound on with a plain gray frame.
3. **Test 4, tone-shift timing.** One 1000 Hz tone, 0.5 seconds long,
   placed at about 2, 6, and 10 seconds in 12 second files. Presence,
   early-or-late, and the numeric start are asked separately.

**The feed proof rule.** Before every answer is taken, the runner
prints the proof for that exact feed: both ffmpeg return codes, the
extracted track duration, sample rate, sample count, overall RMS in
dB, nonzero share, the unit count, every one-second unit's RMS, the
zero-chunk count, any padding samples, and a hash of the exact decoded
track handed toward the model. A copy of each unique extracted track
is also saved beside the results. If a track that should carry sound
decodes missing, empty, short, or all zeros, the run hard-stops with
a loud banner. Declared silence and declared muted versions are
exempt by name in the job list, never by accident. Nothing is padded
quietly: a short final chunk is padded only to complete its second,
and the padded sample count is printed in that ask's proof.

**The yes-score fact, settled from the code before any run.**
MiniCPM's `model.chat` does not expose a yes-token score. Reading
`modeling_minicpmo.py`: `chat` calls `generate`, `_decode` asks the
language model for sequences and hidden states but never requests
output scores, and `chat` returns only decoded text (or an
`OmniOutput` holding text). So every row's yes-score column reads
`not exposed`, and the plain yes or no words are the record. No
internal path is substituted, because that would no longer be the
chat door being tested.

**How to run it.** Run the cells from the top, in order. Step 1
builds the same private environment as the MiniCPM Series One
notebook and proves the stack. Step 2 gets the seven swap files,
builds the gray-frame versions and the three tone files, measures
them from their decoded tracks, and saves the new files into the
Drive clips folder so OLA can ride the identical cells later. Step 3
writes the runner and runs its check mode (proof, loading ladder,
sanity). Step 4 runs the diagnostic battery and saves
`series2_minicpm_results.csv` in the Drive clips folder.
"""))
cells.append(step1_md)
cells.append(step1_code)

# ---------------------------------------------------------------- step 2
cells.append(md(r"""
## Step 2 · Get the swap files and build the new versions

The seven swap files are already in the Drive folder named
`project-k-clips`: `swap_silence.mp4`, `swap_speech.mp4`,
`swap_beeps.mp4`, `swap_water.mp4`, `swap_vacuum.mp4`,
`swap_sizzle.mp4`, and `swap_mixed.mp4`. One still picture, seven
soundtracks, each 10 seconds, loudness matched at build time to
about RMS -20 dB (silence is true digital silence).

This cell also builds two new families of files, measures each one
from its decoded track, and saves them into the same Drive clips
folder so a later model (OLA first) can ride identical cells:

1. **Gray-frame versions** (`series2_gray_<sound>.mp4`): each swap
   soundtrack over a plain gray frame, same size and length as the
   swap still versions.
2. **Tone files** (`series2_tone_2s.mp4`, `series2_tone_6s.mp4`,
   `series2_tone_10s.mp4`): the swap still picture for 12 seconds,
   carrying one 1000 Hz tone, 0.5 seconds long, the same single tone
   the probe file uses, placed at about 2, 6, and 10 seconds. Each
   file's total track level is matched to RMS -20 dB by a measured
   gain loop, then the finished MP4 is decoded back and its tone
   placement is measured and printed.

Do not go on until all seven swap files say ready and the build
table prints a measured placement for each tone file.
"""))

cells.append(code(r"""
# --- Get the swap files; build and measure the gray and tone versions ----------
import os, shutil, subprocess, json, math, wave, hashlib
import numpy as np

SWAP = ["silence", "speech", "beeps", "water", "vacuum", "sizzle", "mixed"]
WORK = "/content/clips"
DRIVE_CLIPS = "/content/drive/MyDrive/project-k-clips"
os.makedirs(WORK, exist_ok=True)

try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception as e:
    print("Drive mount skipped:", e)

SOURCES = [DRIVE_CLIPS, "/content/project-k-clips"]

for sound in SWAP:
    name = f"swap_{sound}.mp4"
    dest = os.path.join(WORK, name)
    if os.path.exists(dest):
        continue
    for folder in SOURCES:
        src = os.path.join(folder, name)
        if os.path.exists(src):
            shutil.copy(src, dest)
            print("copied", name, "from", folder)
            break

missing = [s for s in SWAP
           if not os.path.exists(os.path.join(WORK, f"swap_{s}.mp4"))]
if missing:
    print("MISSING swap files:", missing)
    raise SystemExit("STOP: swap files missing; do not go on")

def ffprobe_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path], capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except Exception:
        return None

def decode_mono(path, sr=48000):
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1",
         "-ar", str(sr), "-f", "f32le", "-"],
        capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"decode failed for {path}: "
                           + proc.stderr.decode(errors="replace")[-500:])
    return np.frombuffer(proc.stdout, dtype=np.float32), sr

def rms_db(x):
    if len(x) == 0:
        return -120.0
    v = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    return 20.0 * math.log10(v) if v > 0 else -120.0

def save_both(local_path):
    # Keep the work copy and, when Drive is mounted, the clips-folder
    # copy that later models will ride.
    if os.path.isdir(DRIVE_CLIPS):
        shutil.copy(local_path, os.path.join(DRIVE_CLIPS,
                                             os.path.basename(local_path)))
        print("saved to Drive clips folder:", os.path.basename(local_path))

build_log = []

# ---- gray-frame versions -------------------------------------------------------
for sound in SWAP:
    out = os.path.join(WORK, f"series2_gray_{sound}.mp4")
    if not os.path.exists(out):
        src = os.path.join(WORK, f"swap_{sound}.mp4")
        proc = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
             "-i", "color=c=0x808080:s=854x480:r=25:d=10",
             "-i", src, "-map", "0:v", "-map", "1:a", "-t", "10",
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-ar", "48000", "-shortest", out])
        if proc.returncode != 0 or not os.path.exists(out):
            raise SystemExit(f"STOP: gray build failed for {sound}")
    x, sr = decode_mono(out)
    entry = {"file": os.path.basename(out),
             "duration_s": round(len(x) / sr, 3),
             "audio_rms_db_decoded": round(rms_db(x), 2),
             "nonzero_share": round(float(np.count_nonzero(x) / len(x)), 4)}
    build_log.append(entry)
    save_both(out)
    print("gray:", entry)

# ---- tone files ----------------------------------------------------------------
still = os.path.join(WORK, "series2_still.png")
subprocess.run(["ffmpeg", "-y", "-v", "error",
                "-i", os.path.join(WORK, "swap_speech.mp4"),
                "-frames:v", "1", still], check=True)

TONE_SR = 48000
TONE_DUR = 12.0
TARGET_RMS = -20.0

def write_tone_wav(path, start_s, amplitude):
    n = int(TONE_SR * TONE_DUR)
    x = np.zeros(n, dtype=np.float64)
    m = int(TONE_SR * 0.5)
    s = int(round(start_s * TONE_SR))
    t = np.arange(m) / TONE_SR
    x[s:s + m] = amplitude * np.sin(2.0 * np.pi * 1000.0 * t)
    pcm = np.clip(np.round(x * 32767), -32768, 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(TONE_SR)
        w.writeframes(pcm.tobytes())

def measure_tone(path):
    x, sr = decode_mono(path, TONE_SR)
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    thr = max(0.03, peak * 0.05)
    idx = np.where(np.abs(x) > thr)[0]
    onset = float(idx[0] / sr) if len(idx) else None
    return {"duration_s": round(len(x) / sr, 3),
            "audio_rms_db_decoded": round(rms_db(x), 2),
            "peak": round(peak, 3),
            "measured_onset_s": None if onset is None else round(onset, 3),
            "nonzero_share": round(float(np.count_nonzero(x) / len(x)), 4)}

for start_s in (2, 6, 10):
    out = os.path.join(WORK, f"series2_tone_{start_s}s.mp4")
    wav_path = os.path.join(WORK, f"series2_tone_{start_s}s.wav")
    amplitude = 0.6928
    entry = None
    for attempt in range(6):
        write_tone_wav(wav_path, start_s, amplitude)
        proc = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-loop", "1", "-framerate", "25",
             "-i", still, "-i", wav_path, "-t", "12",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "25",
             "-c:a", "aac", "-ar", str(TONE_SR), "-shortest", out])
        if proc.returncode != 0 or not os.path.exists(out):
            raise SystemExit(f"STOP: tone build failed for {start_s}s")
        entry = measure_tone(out)
        if abs(entry["audio_rms_db_decoded"] - TARGET_RMS) <= 0.7:
            break
        amplitude *= 10 ** ((TARGET_RMS - entry["audio_rms_db_decoded"]) / 20.0)
        if amplitude >= 0.98:
            raise SystemExit(f"STOP: tone gain loop ran away for {start_s}s")
    entry = {"file": os.path.basename(out), "target_start_s": start_s,
             **entry, "amplitude": round(float(amplitude), 4)}
    build_log.append(entry)
    save_both(out)
    print("tone:", entry)

log_path = os.path.join(WORK, "series2_build_log.json")
with open(log_path, "w") as fh:
    json.dump(build_log, fh, indent=2)
if os.path.isdir(DRIVE_CLIPS):
    shutil.copy(log_path, os.path.join(DRIVE_CLIPS,
                                       "series2_build_log.json"))
print()
print("ready: all 7 swap files, 7 gray versions, 3 tone files")
print("build log:", log_path)
"""))

# ---------------------------------------------------------------- runner
RUNNER_SRC = r"""
# minicpm_series2_run.py - the Series Two diagnostic runner for MiniCPM-o 2.6.
# Written by project_k_minicpm_series2_colab.ipynb. Run only under the
# private environment's Python (/content/minicpm-venv/bin/python).
# Modes: "check" runs the proof, the loading ladder, and the sanity
# question, then stops. "battery" does all of that and then runs the
# Series Two diagnostic tests and saves the CSV.
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else "check"
print("runner mode:", MODE)
print("runner python:", sys.version.split()[0], sys.executable)

# --- Proof lines, printed before anything loads --------------------------------
import torch, transformers
print("torch version:", torch.__version__)
print("transformers version:", transformers.__version__)
print("transformers file:", transformers.__file__)
from transformers.models.whisper.modeling_whisper import WHISPER_ATTENTION_CLASSES
print("WHISPER_ATTENTION_CLASSES import: OK",
      sorted(WHISPER_ATTENTION_CLASSES.keys()))
if transformers.__version__ != "4.44.2":
    print("STOP: this script is running on transformers",
          transformers.__version__, "and needs exactly 4.44.2.")
    sys.exit(3)

# The yes-score question, settled from the model's own code before the
# run (modeling_minicpmo.py in the pinned revision): MiniCPMO.chat
# calls generate, _decode requests sequences and hidden states from
# llm.generate but never output scores, and chat returns only the
# decoded text. There is no score surface on the chat door, so this
# constant is False and every CSV row marks the score as not exposed.
YES_SCORE_EXPOSED = False
print("yes token score exposed by model.chat:", YES_SCORE_EXPOSED,
      "(chat returns decoded text only; output scores are never requested)")

# --- Load the model: the same MiniCPM-o ladder as Series One --------------------
import traceback, shutil, datetime
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

REPO = "openbmb/MiniCPM-o-2_6"
REVISION = "06849bfd36da94da1f5a88fa17e8ba63f08a4c4d"

LOAD_LOG_PATHS = ["/content/drive/MyDrive/project-k-clips/series2_load_log.txt",
                  "/content/series2_load_log.txt"]

def log_load(text):
    for p in LOAD_LOG_PATHS:
        try:
            with open(p, "a") as fh:
                fh.write(text)
        except Exception:
            pass

def log_paths_saved():
    return [p for p in LOAD_LOG_PATHS if os.path.exists(p)]

log_load("=" * 70 + "\n")
log_load("MiniCPM-o 2.6 Series Two load log - "
         + datetime.datetime.now().isoformat(timespec="seconds")
         + " - runner mode: " + MODE + "\n")
log_load("repo: " + REPO + " at pinned revision " + REVISION + "\n")

try:
    from transformers.dynamic_module_utils import HF_MODULES_CACHE
    module_cache = os.path.join(str(HF_MODULES_CACHE), *REPO.split("/"))
    if os.path.isdir(module_cache):
        shutil.rmtree(module_cache)
        clear_note = "cleared cached remote code for this repo: " + module_cache
    else:
        clear_note = ("no cached remote code for this repo to clear: "
                      + module_cache)
except Exception as e:
    clear_note = "WARNING: the remote-code cache clear did not run: " + str(e)
print(clear_note)
log_load(clear_note + "\n")

def free_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

model = None
tokenizer = None
MODEL_NAME = None
LOAD_MODE = None

def try_load(mode):
    kwargs = dict(trust_remote_code=True, attn_implementation="sdpa",
                  init_vision=True, init_audio=True, init_tts=False)
    if mode == "bf16":
        m = AutoModel.from_pretrained(
            REPO, revision=REVISION, torch_dtype=torch.bfloat16, **kwargs)
        m = m.eval().cuda()
    elif mode == "8-bit":
        m = AutoModel.from_pretrained(
            REPO, revision=REVISION, torch_dtype=torch.bfloat16, load_in_8bit=True,
            device_map="cuda", **kwargs)
        m = m.eval()
    elif mode == "4-bit":
        m = AutoModel.from_pretrained(
            REPO, revision=REVISION, torch_dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16),
            device_map="cuda", **kwargs)
        m = m.eval()
    t = AutoTokenizer.from_pretrained(REPO, revision=REVISION,
                                      trust_remote_code=True)
    return m, t

rung_errors = []
for mode in ("bf16", "8-bit", "4-bit"):
    print("=" * 70)
    print(f"rung: {REPO} in {mode} form")
    try:
        m, t = try_load(mode)
        model, tokenizer = m, t
        MODEL_NAME, LOAD_MODE = REPO, mode
        print("loaded:", REPO, "in", mode, "form")
        break
    except Exception:
        tb = traceback.format_exc()
        rung_errors.append((REPO, mode, tb))
        print("this rung failed. Full traceback:")
        print(tb)
        log_load("-" * 70 + "\n")
        log_load("rung failed: " + REPO + " in " + mode + " form\n")
        log_load(tb)
        model = None
        tokenizer = None
        free_memory()

print()
if model is None:
    print("!" * 70)
    print("ALL THREE RUNGS FAILED. MiniCPM-o 2.6 did not load in any fit.")
    saved = log_paths_saved()
    if saved:
        print("The full traceback of every rung is saved, word for word, in:")
        for p in saved:
            print("   ", p)
    else:
        print("The load log could not be written anywhere; the full")
        print("tracebacks are in the output above, and nowhere else.")
    print("Do not go on. Show this whole output to Muse.")
    print("!" * 70)
    sys.exit(4)
print("MODEL FOR THIS RUN:", MODEL_NAME, "in", LOAD_MODE, "form")
log_load("loaded: " + MODEL_NAME + " in " + LOAD_MODE + " form\n")
if torch.cuda.is_available():
    used = torch.cuda.memory_allocated() / 1e9
    print(f"chip memory already used by the model: {used:.1f} GB")

# --- Sanity check: can the model write at all? ---------------------------------
def answer_text(res):
    if isinstance(res, dict):
        return str(res.get("text", "") or "")
    if isinstance(res, (list, tuple)):
        return answer_text(res[0]) if res else ""
    return str(res or "")

try:
    msgs = [{"role": "user",
             "content": ["Reply with the word ready."]}]
    sres = model.chat(msgs=msgs, tokenizer=tokenizer,
                      sampling=False, max_new_tokens=16)
    sraw = answer_text(sres).strip()
    print("SANITY RAW OUTPUT:", repr(sraw))
    if sraw:
        print("sanity check passed: the model can write.")
    else:
        print("WARNING: the sanity check came back EMPTY.")
        print("STOP. Show this output to Muse.")
        sys.exit(5)
except SystemExit:
    raise
except Exception as e:
    print("WARNING: the sanity check FAILED with an error:")
    print(str(e))
    print("STOP. Show this output to Muse.")
    sys.exit(5)

if MODE == "check":
    print()
    print("check mode done: proof, ladder, and sanity all passed.")
    print("The diagnostic battery was not run. Run Step 4 for the tests.")
    sys.exit(0)
"""

RUNNER_SRC += r"""
# --- Diagnostic battery ---------------------------------------------------------
import csv, glob, hashlib, json, math, re, subprocess
import numpy as np
import librosa
import soundfile as sf
from PIL import Image

# The repaired JSON question, injected at build time from
# pilot2/run_eggs.py, character for character (444 characters).
PROMPT_JSON = __PROMPT_JSON__
assert len(PROMPT_JSON) == 444, f"prompt drifted: {len(PROMPT_JSON)} chars"

CLIPS = "/content/clips"
TRACK_DIRS = ["/content/drive/MyDrive/project-k-clips/series2_extracted_tracks",
              "/content/series2_extracted_tracks"]
for d in TRACK_DIRS:
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass

FEED_WITH_PICTURE = ("omni chat units: 1 frame per second + the matching 1s "
                     "of the 16 kHz audio track per unit, frames downscaled "
                     "to fit 448x448, max 15 units, max_slice_nums=1, "
                     "greedy (sampling=False), no generated speech")
FEED_AUDIO_ALONE = ("closest audio-only unified chat form: model.chat "
                    "accepts numpy audio with no image; the code has no "
                    "separate audio-only method, so the unified chat is "
                    "used with the default audio-and-text system prompt, "
                    "per-second 16 kHz audio chunks, no picture and no "
                    "unit markers, greedy (sampling=False), "
                    "no generated speech")

class FeedProofError(RuntimeError):
    pass

def rms_db(x):
    if len(x) == 0:
        return -120.0
    v = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    return 20.0 * math.log10(v) if v > 0 else -120.0

def media_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path], capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except Exception:
        return None

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

FRAME_DIR = "/content/series2_frames"
os.makedirs(FRAME_DIR, exist_ok=True)
_feed_cache = {}

def extract_feed(path, with_picture, expected_sound, expected_silence):
    key = (path, bool(with_picture), bool(expected_sound),
           bool(expected_silence))
    if key in _feed_cache:
        return _feed_cache[key]
    if not os.path.exists(path):
        raise FeedProofError(f"feed file missing: {path}")
    tag = re.sub(r"[^A-Za-z0-9]+", "_",
                 os.path.basename(path) + "_" + str(with_picture))
    fdir = os.path.join(FRAME_DIR, tag)
    os.makedirs(fdir, exist_ok=True)
    for old in glob.glob(os.path.join(fdir, "*")):
        os.remove(old)

    frame_rc = None
    frames = []
    if with_picture:
        fproc = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", path, "-vf", "fps=1",
             os.path.join(fdir, "f%03d.png")], capture_output=True, text=True)
        frame_rc = fproc.returncode
        frames = sorted(glob.glob(os.path.join(fdir, "*.png")))
        if frame_rc != 0:
            raise FeedProofError(
                f"frame extraction failed for {path}: rc={frame_rc}: "
                + (fproc.stderr or "")[-500:])
        if not frames:
            raise FeedProofError(f"frame extraction produced no frames: {path}")

    wav = os.path.join(fdir, "audio.wav")
    aproc = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", path, "-vn", "-ac", "1",
         "-ar", "16000", wav], capture_output=True, text=True)
    audio_rc = aproc.returncode
    if audio_rc != 0:
        raise FeedProofError(
            f"audio extraction failed for {path}: rc={audio_rc}: "
            + (aproc.stderr or "")[-500:])
    if not os.path.exists(wav):
        raise FeedProofError(f"audio extraction produced no wav for {path}")
    audio_np, sr = librosa.load(wav, sr=16000, mono=True)
    audio_np = np.asarray(audio_np, dtype=np.float32)

    src_dur = media_duration(path)
    if src_dur is None:
        raise FeedProofError(f"could not read media duration for {path}")
    unit_count = int(round(src_dur))
    if unit_count <= 0:
        raise FeedProofError(f"media duration rounds to zero units: {path}")
    if unit_count > 15:
        unit_count = 15
    track_dur = len(audio_np) / sr if sr else 0.0
    if len(audio_np) == 0:
        raise FeedProofError(f"extracted track is empty for {path}")
    if track_dur < unit_count - 0.5:
        raise FeedProofError(
            f"extracted track is short for {path}: {track_dur:.3f}s of "
            f"track for {unit_count} units")

    if with_picture and len(frames) < unit_count - 1:
        raise FeedProofError(
            f"frame extraction is short for {path}: {len(frames)} frames "
            f"for {unit_count} units")

    per_unit_rms = []
    per_unit_padding = []
    zero_chunks = 0
    for i in range(unit_count):
        chunk = audio_np[sr * i: sr * (i + 1)]
        if len(chunk) == 0:
            raise FeedProofError(
                f"unit {i} has no audio samples for {path}")
        per_unit_rms.append(round(rms_db(chunk), 2))
        if not np.any(chunk):
            zero_chunks += 1
        per_unit_padding.append(int(sr - len(chunk)) if len(chunk) < sr else 0)

    overall_rms = round(rms_db(audio_np), 2)
    nonzero_share = float(np.count_nonzero(audio_np) / len(audio_np))
    if expected_sound and (nonzero_share == 0.0 or overall_rms <= -119.5
                           or zero_chunks == unit_count):
        raise FeedProofError(
            f"track that should carry sound decoded silent for {path}: "
            f"rms={overall_rms} dB, nonzero share={nonzero_share}")
    if expected_silence and nonzero_share > 0.001:
        raise FeedProofError(
            f"declared silence track is not silent for {path}: "
            f"nonzero share={nonzero_share}")

    track_hash = hashlib.sha256(audio_np.tobytes()).hexdigest()
    track_name = (re.sub(r"[^A-Za-z0-9]+", "_", os.path.basename(path))
                  + "__" + track_hash[:12] + ".wav")
    track_paths = []
    for d in TRACK_DIRS:
        try:
            tp = os.path.join(d, track_name)
            if not os.path.exists(tp):
                sf.write(tp, audio_np, sr, subtype="PCM_16")
            track_paths.append(tp)
        except Exception:
            pass

    contents = []
    for i in range(unit_count):
        chunk = audio_np[sr * i: sr * (i + 1)]
        if len(chunk) < sr:
            chunk = np.pad(chunk, (0, sr - len(chunk))).astype(np.float32)
        if with_picture:
            img = Image.open(frames[min(i, len(frames) - 1)]).convert("RGB")
            img.thumbnail((448, 448))
            contents.extend(["<unit>", img, chunk])
        else:
            contents.append(chunk)

    proof = {
        "file": os.path.basename(path),
        "source_file_sha256": file_sha256(path),
        "frame_ffmpeg_rc": frame_rc,
        "audio_ffmpeg_rc": audio_rc,
        "track_duration_s": round(track_dur, 3),
        "sample_rate": int(sr),
        "sample_count": int(len(audio_np)),
        "track_rms_db": overall_rms,
        "track_nonzero_share": round(nonzero_share, 6),
        "unit_count": int(unit_count),
        "per_unit_rms_db": per_unit_rms,
        "per_unit_padding_samples": per_unit_padding,
        "padding_samples_total": int(sum(per_unit_padding)),
        "zero_chunk_count": int(zero_chunks),
        "track_sha256": track_hash,
        "track_copy_paths": track_paths,
        "expected_sound": bool(expected_sound),
        "expected_silence": bool(expected_silence),
        "feed": FEED_WITH_PICTURE if with_picture else FEED_AUDIO_ALONE,
    }
    feed = {"contents": contents, "proof": proof}
    _feed_cache[key] = feed
    return feed

def ask(job, prompt, max_new_tokens):
    with_picture = job["arm"] != "audio_alone"
    feed = extract_feed(os.path.join(CLIPS, job["file"]), with_picture,
                        job["expected_sound"], job["expected_silence"])
    proof = feed["proof"]
    print("FEED PROOF " + json.dumps(proof, sort_keys=True))
    contents = list(feed["contents"]) + [prompt]
    if job["arm"] == "audio_alone":
        sys_msg = model.get_sys_prompt(mode="default", language="en")
        res = model.chat(msgs=[sys_msg, {"role": "user", "content": contents}],
                         tokenizer=tokenizer, sampling=False,
                         max_new_tokens=max_new_tokens, omni_input=False,
                         chunk_input=True, use_tts_template=False,
                         generate_audio=False, max_slice_nums=1,
                         use_image_id=False)
    else:
        sys_msg = model.get_sys_prompt(mode="omni", language="en")
        res = model.chat(msgs=[sys_msg, {"role": "user", "content": contents}],
                         tokenizer=tokenizer, sampling=False,
                         max_new_tokens=max_new_tokens, omni_input=True,
                         use_tts_template=False, generate_audio=False,
                         max_slice_nums=1, use_image_id=False)
    text = answer_text(res).strip()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return text, proof

# --- Test definitions ------------------------------------------------------------
SWAP_FILES = {
    "silence": "swap_silence.mp4",
    "speech": "swap_speech.mp4",
    "beeps": "swap_beeps.mp4",
    "water": "swap_water.mp4",
    "vacuum": "swap_vacuum.mp4",
    "sizzle": "swap_sizzle.mp4",
    "mixed": "swap_mixed.mp4",
}

PRESENCE_PROMPTS = {
    "speech": "Is there a person speaking in this clip? Answer yes or no.",
    "silence": "Is this clip silent? Answer yes or no.",
    "water": "Is there the sound of running water in this clip? Answer yes or no.",
    "beeps": "Is there a beeping sound in this clip? Answer yes or no.",
}
DETAIL_PROMPTS = {
    "speech": "Write down any words spoken in this clip.",
    "silence": "Is there any sound at all in this clip? Answer yes or no.",
    "water": "What sound, if any, do you hear in this clip? Answer with a short sound name or none.",
    "beeps": "What sound, if any, do you hear in this clip? Answer with a short sound name or none.",
}

def base_job(test, sound, file, arm, version, expected_sound):
    return {"test": test, "sound": sound, "file": file, "arm": arm,
            "version": version,
            "expected_sound": bool(expected_sound),
            "expected_silence": not bool(expected_sound)}

JOBS = []

# Test 2: the 2x2. Speech baseline first, silence as the negative
# control, then water and beeps. Arms: audio alone, then the same
# sound with the swap still picture. Forms: open, then the two direct
# asks (presence, and detail: transcription for speech, a short sound
# name or the any-sound control for the others). Two repeats per cell.
for sound in ("speech", "silence", "water", "beeps"):
    expected = sound != "silence"
    for arm, version in (("audio_alone", "no_picture"),
                         ("audio_with_still", "original_swap_still")):
        for rep in (1, 2):
            job = base_job("test2_2x2", sound, SWAP_FILES[sound], arm,
                           version, expected)
            JOBS.append((job, "open", "open_json", sound, PROMPT_JSON,
                         rep, 300))
            job = base_job("test2_2x2", sound, SWAP_FILES[sound], arm,
                           version, expected)
            JOBS.append((job, "direct", "presence", sound,
                         PRESENCE_PROMPTS[sound], rep, 32))
            job = base_job("test2_2x2", sound, SWAP_FILES[sound], arm,
                           version, expected)
            label = "transcription" if sound == "speech" else "detail"
            JOBS.append((job, "direct", label, sound,
                         DETAIL_PROMPTS[sound], rep, 128))

# Test 3: yes/no scoring on the full swap set. Three versions per
# sound: sound on with the still picture, muted with the still
# picture (the true silence file, declared in the job), and sound on
# with a plain gray frame. Matching target per file, plus the two
# lures: the speech file is also asked about water, and the water
# file is also asked about speech. The mixed file carries two sounds,
# so it gets both of its matching targets.
TARGETS = {
    "silence": ["silence"],
    "speech": ["a person speaking"],
    "beeps": ["beeping"],
    "water": ["running water"],
    "vacuum": ["a vacuum cleaner"],
    "sizzle": ["sizzling"],
    "mixed": ["a person speaking", "a vacuum cleaner"],
}
LURES = {"speech": ["running water"], "water": ["a person speaking"]}
for sound in ("silence", "speech", "beeps", "water", "vacuum",
              "sizzle", "mixed"):
    for version in ("sound_on_still", "muted_still", "sound_on_gray"):
        if version == "sound_on_still":
            file = SWAP_FILES[sound]
            expected = sound != "silence"
        elif version == "muted_still":
            file = SWAP_FILES["silence"]
            expected = False
        else:
            file = f"series2_gray_{sound}.mp4"
            expected = sound != "silence"
        for target, label in ([(t, "matching_yes_no") for t in TARGETS[sound]]
                              + [(t, "lure_yes_no")
                                 for t in LURES.get(sound, [])]):
            job = base_job("test3_yes_no", sound, file, "audio_with_frame",
                           version, expected)
            prompt = (f"Is there the sound of {target} in this clip? "
                      "Answer yes or no.")
            JOBS.append((job, "yes_no", label, target, prompt, 1, 16))

# Test 4: tone-shift timing. Three files, one tone placed at about 2,
# 6, and 10 seconds. Three separate asks per file: presence,
# early-or-late, numeric. Timing is scored later, and only where
# presence was detected; the notebook records the raw answers.
for start_s in (2, 6, 10):
    file = f"series2_tone_{start_s}s.mp4"
    for form, label, prompt, tokens in (
            ("presence", "presence",
             "Is there a tone in this clip? Answer yes or no.", 16),
            ("early_late", "early_late",
             "Does the tone start in the first half or the second half "
             "of the clip?", 32),
            ("numeric", "numeric_start",
             "At what second does the tone start? Answer with a number.",
             32)):
        job = base_job("test4_tone_shift", "tone", file, "audio_with_still",
                       "tone_shift_still", True)
        job["target_start_s"] = start_s
        JOBS.append((job, form, label, "tone", prompt, 1, tokens))

print(f"diagnostic jobs: {len(JOBS)} asks "
      f"(test 2: 48, test 3: {sum(1 for j in JOBS if j[0]['test'] == 'test3_yes_no')}, "
      f"test 4: 9)")

FIELDS = ["test", "sound", "file", "arm", "version", "question_form",
          "question_label", "target", "prompt", "repeat",
          "target_start_s", "expected_sound", "expected_silence",
          "frame_ffmpeg_rc", "audio_ffmpeg_rc", "track_duration_s",
          "sample_rate", "sample_count", "track_rms_db",
          "track_nonzero_share", "unit_count", "per_unit_rms_db",
          "per_unit_padding_samples", "padding_samples_total",
          "zero_chunk_count", "track_sha256", "track_copy_paths",
          "source_file_sha256", "raw_answer", "yes_score",
          "yes_score_status", "model", "feed"]

rows = []
failed = 0

def save_rows():
    out_fields = FIELDS
    for out_path in ("/content/drive/MyDrive/project-k-clips/series2_minicpm_results.csv",
                     "/content/series2_minicpm_results.csv"):
        try:
            with open(out_path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=out_fields)
                w.writeheader()
                w.writerows(rows)
            print("saved:", out_path)
        except Exception as e:
            print("could not save", out_path, "-", str(e)[:120])

for job, form, label, target, prompt, rep, max_tokens in JOBS:
    print("=" * 70)
    print(f"{job['test']} - sound={job['sound']} arm={job['arm']} "
          f"version={job['version']} form={form}/{label} "
          f"target={target} repeat={rep}")
    raw_text = ""
    error = ""
    proof = None
    try:
        raw, proof = ask(job, prompt, max_tokens)
        if not raw.strip():
            error = "ERROR: empty answer returned"
            failed += 1
            print("WARNING: this ask came back EMPTY. Recorded as an error,")
            print("never as silence.")
            raw_text = error
        else:
            raw_text = raw
            print("RAW ANSWER:")
            print(raw)
    except FeedProofError as e:
        print()
        print("!" * 70)
        print("FEED PROOF HARD STOP. The feed for this ask failed its proof:")
        print(str(e))
        print("No answer was taken for this ask, and the run stops here.")
        print("Rows completed before this ask are saved now. Show this")
        print("whole output to Muse.")
        print("!" * 70)
        save_rows()
        sys.exit(7)
    except Exception as e:
        error = "ERROR: " + traceback.format_exc()
        failed += 1
        print(f"WARNING: this ask FAILED with an error. The full error is")
        print("saved in the CSV column raw_answer.")
        raw_text = error
        proof = proof or {}
    p = proof or {}
    rows.append({
        "test": job["test"], "sound": job["sound"], "file": job["file"],
        "arm": job["arm"], "version": job["version"],
        "question_form": form, "question_label": label, "target": target,
        "prompt": prompt, "repeat": rep,
        "target_start_s": job.get("target_start_s", ""),
        "expected_sound": job["expected_sound"],
        "expected_silence": job["expected_silence"],
        "frame_ffmpeg_rc": p.get("frame_ffmpeg_rc", ""),
        "audio_ffmpeg_rc": p.get("audio_ffmpeg_rc", ""),
        "track_duration_s": p.get("track_duration_s", ""),
        "sample_rate": p.get("sample_rate", ""),
        "sample_count": p.get("sample_count", ""),
        "track_rms_db": p.get("track_rms_db", ""),
        "track_nonzero_share": p.get("track_nonzero_share", ""),
        "unit_count": p.get("unit_count", ""),
        "per_unit_rms_db": json.dumps(p.get("per_unit_rms_db", "")),
        "per_unit_padding_samples": json.dumps(
            p.get("per_unit_padding_samples", "")),
        "padding_samples_total": p.get("padding_samples_total", ""),
        "zero_chunk_count": p.get("zero_chunk_count", ""),
        "track_sha256": p.get("track_sha256", ""),
        "track_copy_paths": json.dumps(p.get("track_copy_paths", "")),
        "source_file_sha256": p.get("source_file_sha256", ""),
        "raw_answer": raw_text,
        "yes_score": "not exposed",
        "yes_score_status": ("model.chat returns decoded text only and "
                             "does not request output scores"),
        "model": (f"{MODEL_NAME} ({LOAD_MODE}) "
                  "[venv py3.12, torch 2.8.0+cu126]"),
        "feed": p.get("feed", ""),
    })
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if failed:
    print()
    print("!" * 70)
    print(f"RUN PROBLEM: {failed} ask(s) failed or came back empty. Their")
    print("raw words or full errors are in the CSV column raw_answer.")
    print("This run is NOT clean evidence until those asks are explained.")
    print("!" * 70)
else:
    print()
    print("Clean run: every ask answered, every feed proved in print first.")

save_rows()
if failed:
    sys.exit(6)
"""

RUNNER_SRC = RUNNER_SRC.replace("__PROMPT_JSON__", repr(PROMPT_JSON))
assert "__PROMPT_" not in RUNNER_SRC, "prompt placeholders survived"

# ---------------------------------------------------------------- step 3
cells.append(md(r"""
## Step 3 · Write the runner, then check: ladder and sanity

The first cell below writes the runner script,
`/content/minicpm_series2_run.py`. It holds the same proof, the same
loading ladder (bf16, then 8-bit, then 4-bit, all pinned to revision
`06849bfd36da94da1f5a88fa17e8ba63f08a4c4d`, remote-code cache cleared
first), and the same sanity gate as the Series One MiniCPM runner.
What changed is only what a failure leaves behind and what the
battery asks: rung tracebacks now append to `series2_load_log.txt`
in the clips folder, and the battery is the Series Two diagnostic
set, not the twenty Series One questions.

The second cell runs the script in **check mode**: proof lines (now
including the yes-score fact: `model.chat` exposes no score, settled
from the code), the ladder, and the sanity question. If any of those
fail, the script stops and Step 4 must not be run.
"""))

cells.append(code(
    "# --- Write the Series Two runner script ----------------------------------------\n"
    "# The runner holds the ladder, the sanity check, the strict feed\n"
    "# proof, and the diagnostic battery. It is written to a plain file\n"
    "# here and only ever run under the private environment's Python.\n"
    "RUNNER_SOURCE = " + repr(RUNNER_SRC) + "\n"
    "with open(\"/content/minicpm_series2_run.py\", \"w\") as fh:\n"
    "    fh.write(RUNNER_SOURCE)\n"
    "print(\"runner written:\", \"/content/minicpm_series2_run.py\", "
    "len(RUNNER_SOURCE), \"chars\")\n"))

cells.append(code(r"""
# --- Check mode: proof, ladder, sanity -------------------------------------------
import os, subprocess
VENV_PY = "/content/minicpm-venv/bin/python"
proc = subprocess.run([VENV_PY, "/content/minicpm_series2_run.py", "check"])
print()
print("check exit code:", proc.returncode)
if proc.returncode != 0:
    print("!" * 72)
    print("STOP. THE CHECK RUN FAILED. The runner's own output above says")
    print("where: a proof line, a ladder rung, or the sanity question.")
    print("Do not run Step 4. Show this output to Muse.")
    print("!" * 72)
    raise SystemExit("STOP: runner check failed")
print("check passed: proof lines right, model loaded, sanity answered.")
print("Go on to Step 4.")
"""))

# ---------------------------------------------------------------- step 4
cells.append(md(r"""
## Step 4 · Run the diagnostic battery

The next cell runs the runner in **battery mode**: proof and ladder
again, sanity again, then the three tests. Every ask prints its
FEED PROOF line before the model's answer is taken. Watch for it:
that line is the whole point of Series Two.

**Test 2, the 2x2 (48 asks).** Sounds in a fixed order: speech
first (the baseline), then true silence (the negative control),
then water, then beeps. Each sound is fed two ways: as sound alone,
and as the same soundtrack with the swap still picture. The
sound-alone arm uses the model's unified `model.chat` with no
picture: the code has no separate audio-only method, and the
notebook states that difference rather than hiding it. Each arm is
asked three ways, two repeats each: the open repaired JSON question
(word for word the Series One question), a direct presence question
made for that sound, and a direct detail question (for speech:
"Write down any words spoken in this clip."; for the others, a
short sound-name question, and for silence an any-sound control).

**Test 3, yes/no scoring (30 asks).** All seven swap soundtracks,
three versions each: sound on with the still picture, muted with
the still picture, and sound on with a plain gray frame. Each
version is asked about its matching sound ("Is there the sound of
X in this clip? Answer yes or no."), the mixed file about both of
its sounds, and two lures ride along: the speech file is asked
about water, and the water file is asked about speech. The reading
is fixed in advance in the Series Two plan: yes not moving between
sound on and muted means the sound never got in; yes rising while
the free answer still says none means the loss is at reporting;
yes staying high on a muted kitchen picture means the scene is
guessing. Scores are not exposed by `model.chat`, so the yes-score
column reads "not exposed" on every row and the words are the
record.

**Test 4, tone-shift timing (9 asks).** The three tone files from
Step 2. Per file, three separate asks: presence (yes or no),
early-or-late ("Does the tone start in the first half or the
second half of the clip?"), and numeric ("At what second does the
tone start? Answer with a number."). Timing gets scored later,
only on files where presence was detected, so a timing failure
can no longer hide inside a hearing failure.

**Hard stops and warnings.** A feed that fails its proof stops the
whole run on the spot (exit code 7), with the rows completed so
far saved first. An ask that errors or comes back empty is saved
with its full error text in `raw_answer` and a WARNING on screen,
never as silence, and the run ends with the RUN PROBLEM banner
(exit code 6). Results save to `series2_minicpm_results.csv` in
the Drive clips folder, one row per ask, with the full feed-proof
numbers on every row.
"""))

cells.append(code(r"""
# --- Battery mode: the Series Two diagnostic tests --------------------------------
import os, subprocess
VENV_PY = "/content/minicpm-venv/bin/python"
proc = subprocess.run([VENV_PY, "/content/minicpm_series2_run.py", "battery"])
print()
print("battery exit code:", proc.returncode)
if proc.returncode != 0:
    print("!" * 72)
    print("STOP. THE BATTERY RUN DID NOT FINISH CLEAN. Read the runner's")
    print("output above: a gate failed, a feed proof hard-stopped the run,")
    print("or some asks are broken and marked in the CSV. Show it to Muse.")
    print("!" * 72)
    raise SystemExit("STOP: battery run not clean")
print("battery done: every ask answered, every feed proved, CSV saved.")
"""))

# ---------------------------------------------------------------- closing
cells.append(md(r"""
## Closing · How to read the answers

- **Feed proof first.** A row is only as good as its proof line.
  Rows after a hard stop do not exist, on purpose: a stopped run is
  a named failure, not a pile of quiet none answers.
- **Test 2 reads as a square.** Sound alone against sound with a
  picture separates the picture's interference; open against direct
  separates the question's suppression. Speech is the baseline and
  silence is the control: a yes on silence is a false alarm, and
  the rate matters as much as the hits.
- **Test 3 reads by movement.** Compare each sound across its
  three versions before comparing sounds to each other. The
  pattern of what moves, and what does not, is the result.
- **Test 4 reads in order.** Presence first. Early-or-late and
  numeric starts count only where presence was a hit.
- **Scope.** Every claim stays capped at this model, this feed,
  these files, this date. The stages in the Series Two plan stay
  hypotheses until an intervention separates them, and these three
  tests are the interventions.
"""))

nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
        "accelerator": "GPU",
        "colab": {"gpuType": "L4"},
    },
    "cells": cells,
}

out = os.path.join(HERE, "project_k_minicpm_series2_colab.ipynb")
with open(out, "w") as fh:
    json.dump(nb, fh, indent=1)
print("wrote", out, "cells:", len(cells))
