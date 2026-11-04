#!/usr/bin/env python3
"""Build project_k_minicpm_codec_colab.ipynb.

Codec-loss control notebook for MiniCPM-o 2.6. It reuses the
Series Two MiniCPM notebook's plumbing exactly where the two
runs share a path: the same sealed private environment (Step 1
reused byte for byte from the Series One MiniCPM notebook),
the same pinned hub revision, the same loading ladder (bf16,
then 8-bit, then 4-bit), the same feed extraction for files
with a picture (ffmpeg frames at 1 fps, the soundtrack decoded
to a 16 kHz WAV and loaded with librosa, one-second units of
frame plus matching sound chunk through model.chat), the same
feed proof measured on the exact decoded track, and the same
hard stop (exit code 7). The yes-score fact is the Series Two
one: model.chat returns decoded text only, so every row reads
"not exposed" and the words are the record.

What forks, and only this:

1. The battery. 7 sounds x 2 arms x 2 forms x 2 repeats = 56
   asks. (The codec control plan text says 112 beside these
   same factors; the factors multiply to 56, and this build
   follows the factors.) The sounds are the codec-control
   masters built by
   codec-control/build_battery.py. Arm A is the sound inside
   an MP4 (the swap build recipe), fed through the Series Two
   with-picture extraction. Arm B is the same sound as an
   uncompressed WAV: the runner loads the waveform directly
   from the WAV with librosa and builds the same one-second
   units, each carrying the swap still (codec_still.png) and
   the matching chunk of that waveform. The audio array never
   passes through an MP4 or an AAC decode.
2. The forms. Open is the swap study's P3 question, pulled
   from pilot2/swap/run_swap.py at build time, never retyped.
   Yes/no is the Series Two template with the codec targets.
3. There is no model-answer sanity gate. A past gate
   misfired on a missing file; here the file and feed
   assertions are the only gates. Before any ask, every
   battery file is decoded through the very path its arm will
   ride, printed (path, SHA-256, duration, per-second RMS),
   and asserted: duration 10.0 +/- 0.5 s, non-silent unless
   the sound is declared silence, and for Arm B the waveform
   handed toward the model must hash equal to a direct decode
   of the same WAV in the same session. Any failure
   hard-stops with exit 7.
4. The run is resume-safe: asks already present in the
   output CSV are skipped, keyed by (sound, arm, form,
   target, repeat).

The job list is built here, in the builder, and injected
into the runner as a literal, so the count and the prompt
strings are verified at build time, not trusted.

Plain language throughout, no em dashes.
"""
import copy
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT2 = os.path.join(HERE, "..", "pilot2")
BASE_NB = os.path.join(HERE, "project_k_minicpm_o_colab.ipynb")


def extract_p3(path):
    src = open(path).read()
    m = re.search(r"'P3': \((.*?)\)\s*,\n", src, re.S)
    if not m:
        raise SystemExit(f"could not find P3 in {path}")
    ns = {}
    exec("P3 = (" + m.group(1) + ")", ns)
    return ns["P3"]


P3 = extract_p3(os.path.join(PILOT2, "swap", "run_swap.py"))
P3_EXPECTED = ("What sounds can you hear in this clip? Describe only "
               "what is audible, without guessing from the pictures. "
               "If the audio is silent, say 'silent'.")
assert P3 == P3_EXPECTED, f"P3 drifted: {P3!r}"

# ---------------------------------------------------------------- battery spec
SOUNDS = ["tone_ref", "tone_quiet", "clicks_quiet", "speech",
          "tone_under_bed", "burst_quiet", "silence"]
YESNO_TARGETS = {
    "tone_ref": "a tone",
    "tone_quiet": "a tone",
    "tone_under_bed": "a tone",
    "clicks_quiet": "clicking",
    "speech": "speech",
    "burst_quiet": "sizzling",
    "silence": "a tone",
}


def yesno_prompt(target):
    return (f"Is there the sound of {target} in this clip? "
            "Answer yes or no.")


JOBS = []
for sound in SOUNDS:
    for arm, version in (("aac_mp4", "codec_arm_a"),
                         ("raw_wav", "codec_arm_b")):
        file = (f"codec_{sound}_aac.mp4" if arm == "aac_mp4"
                else f"codec_{sound}_raw.wav")
        expected = sound != "silence"
        for form, label, tokens in (("open", "open_p3", 300),
                                    ("yes_no", "matching_yes_no", 16)):
            if form == "open":
                target, prompt = sound, P3
            else:
                target = YESNO_TARGETS[sound]
                prompt = yesno_prompt(target)
            for rep in (1, 2):
                job = {"test": "codec_control", "sound": sound,
                       "file": file, "arm": arm, "version": version,
                       "expected_sound": bool(expected),
                       "expected_silence": not bool(expected)}
                JOBS.append((job, form, label, target, prompt, rep,
                             tokens))
assert len(JOBS) == 56, f"battery size drifted: {len(JOBS)}"


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
# Project K, codec-loss control: MiniCPM-o 2.6

**Why this notebook exists.** Every Project K run so far fed
the models sound that had passed through a lossy codec: Pilot
One at AAC 96k, Pilot Two and the swap files with the swap
build recipe. A lossy codec throws away quiet detail by
design. So a fair objection stands against the whole record:
when a model missed a quiet sound, maybe the file lost it
before the model ever had the chance. This notebook is the
control that answers the objection. It asks the same model
the same questions about the same sounds, twice over: once
with the sound inside an MP4, decoded the way every earlier
run decoded it, and once with the same sound handed over as
an uncompressed WAV waveform that never passed through the
codec.

**What this notebook does.** Seven sounds, each exactly 10
seconds, each in two arms, each asked two ways, two repeats:
56 asks.

- **The sounds.** A 1000 Hz tone at the reference level
  (burst RMS -20 dBFS); the same tone 20 dB quieter; three
  short sharp clicks at the quiet level; the spoken sentence
  (the swap study's own speech file, so this control ties
  back to it); the quiet tone mixed under the matched vacuum
  bed; a quiet two-second noise burst, the sizzle-shaped
  case; and true silence.
- **Arm A, the file way.** The sound inside an MP4 built
  with the swap recipe: the same still picture, the native
  AAC encoder. The recipe sets no bitrate and the encoder's
  rate follows the content; this battery's speech file
  lands at about 38 kbps, matching the shipped swap speech
  file on the same soundtrack. Arm A reproduces the recipe,
  which is what every earlier run actually rode.
- **Arm B, the raw way.** The same sound as a WAV master.
  The runner loads the waveform straight from the WAV and
  builds the same one-second chat units as Arm A, each unit
  carrying the swap still and the matching second of that
  waveform. The audio array never passes through an MP4.
- **The forms.** The open question is the swap study's P3,
  word for word. The yes/no question is the Series Two form,
  "Is there the sound of X in this clip? Answer yes or no.",
  with X matched to the sound. This model's chat door
  exposes no yes score (settled from the code in Series
  Two), so the yes/no rows are words only, as in its Series
  Two run.

**The gates are files, not answers.** There is no sanity
question in this notebook. A model-answer gate once
misfired on a missing file and stopped a healthy run, so
here the gates are the file and feed assertions, run before
the battery over every file: each file is decoded through
the very path its arm will ride, and the runner prints its
path, SHA-256, duration, and per-second RMS, then asserts
the duration is 10.0 seconds within half a second, asserts
the track is not silent unless the sound is declared
silence (and that declared silence really is zeros), and
for Arm B asserts the waveform handed toward the model
hashes equal to a direct decode of the same WAV in the same
session. Any failure stops the run with exit code 7 before
a single ask. The same proof then repeats per ask, before
each answer, on the exact decoded track the chat receives.

**Resume.** The battery skips any ask already present in
`codec_minicpm_results.csv`, so a stopped session can be
continued by running Step 4 again.

**How to read the result** (fixed in advance, from the
codec control plan):

- Caught raw, missed in AAC: the file did it. That sound's
  earlier misses get re-labeled.
- Missed in both: the model did it; the codec objection
  dies for that sound.
- Caught in both: codec loss does not explain that sound.
- Caught in AAC, missed raw: a build bug. Stop and inspect
  before believing anything.

**Where it runs.** On the paid Colab account (kobbytti),
like the other MiniCPM runs, under its sealed private
environment. **How to run it.** Run the cells from the top,
in order. Step 1 builds the private environment and proves
the stack. Step 2 gets the fifteen battery files. Step 3
writes the runner and runs its check mode (proof, loading
ladder, file assertions). Step 4 runs the battery and saves
`codec_minicpm_results.csv`.

**The one rule.** We claim only what we ran: these sounds,
this model, these two feed paths, this run.
"""))
cells.append(step1_md)
cells.append(step1_code)

# ---------------------------------------------------------------- step 2
cells.append(md(r"""
## Step 2 · Get the codec battery files

Everything this run asks about lives in your Google Drive,
in the folder named `codec-control`. The next cell links
your Drive, finds that folder, and copies fifteen files to
a plain work folder here:

- Seven Arm A files, `codec_<sound>_aac.mp4`: the swap
  still picture carrying each sound, built with the swap
  recipe.
- Seven Arm B masters, `codec_<sound>_raw.wav`: the same
  sounds as uncompressed WAV, 16-bit PCM at 48000 Hz (the
  speech master is stereo, a byte-for-byte copy of the swap
  study's own speech track; the rest are mono).
- `codec_still.png`: the swap still itself, the picture
  the Arm B units carry.

The sounds, in battery order: `tone_ref`, `tone_quiet`,
`clicks_quiet`, `speech`, `tone_under_bed`, `burst_quiet`,
`silence`. Levels and hashes for every file are recorded in
the battery's BATTERY.md; the runner re-measures everything
from inside this session in Step 3 before it believes any
of it.

Do not go on until all fifteen files say "ready".
"""))

cells.append(code(r"""
# --- Get the codec battery files -------------------------------------------------
import os, shutil, subprocess

SOUNDS = ["tone_ref", "tone_quiet", "clicks_quiet", "speech",
          "tone_under_bed", "burst_quiet", "silence"]
FILES = ([f"codec_{s}_aac.mp4" for s in SOUNDS]
         + [f"codec_{s}_raw.wav" for s in SOUNDS]
         + ["codec_still.png"])
WORK = "/content/clips"
DRIVE_CLIPS = "/content/drive/MyDrive/codec-control"
os.makedirs(WORK, exist_ok=True)

try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception as e:
    print("Drive mount skipped:", e)

SOURCES = [DRIVE_CLIPS, "/content/codec-control"]

def bring(name):
    dest = os.path.join(WORK, name)
    if os.path.exists(dest):
        return True
    for folder in SOURCES:
        src = os.path.join(folder, name)
        if os.path.exists(src):
            shutil.copy(src, dest)
            print("copied", name, "from", folder)
            return True
    return False

def duration_s(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path], capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except Exception:
        return None

ready = True
for name in FILES:
    bring(name)
    path = os.path.join(WORK, name)
    if os.path.exists(path):
        d = duration_s(path)
        extra = f", {d:.3f} s" if d is not None else ""
        print("ready:  ", name, f"({os.path.getsize(path) / 1e6:.2f} MB{extra})")
    else:
        ready = False
        print("MISSING:", name)
if not ready:
    print()
    print("!" * 72)
    print("STOP. Some files are missing. Check the Drive folder name is")
    print("exactly codec-control, or upload the files by hand, then")
    print("run this cell again. Do not go on to Step 3.")
    print("!" * 72)
    raise SystemExit("STOP: files missing in Step 2")
print()
print("ready: all 7 Arm A MP4s, all 7 Arm B WAVs, and the still")
"""))

# ---------------------------------------------------------------- runner
RUNNER_SRC = r"""
# minicpm_codec_run.py - the codec-loss control runner for MiniCPM-o 2.6.
# Written by project_k_minicpm_codec_colab.ipynb. Run only under the
# private environment's Python (/content/minicpm-venv/bin/python).
# Modes: "check" runs the proof, the loading ladder, and the battery
# file assertions, then stops. "battery" does all of that and then
# runs the 56 codec asks and saves the CSV.
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

# The yes-score question, settled from the model's own code before
# the run (modeling_minicpmo.py in the pinned revision): MiniCPMO.chat
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

LOAD_LOG_PATHS = ["/content/drive/MyDrive/codec-control/codec_minicpm_load_log.txt",
                  "/content/codec_minicpm_load_log.txt"]

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
log_load("MiniCPM-o 2.6 codec control load log - "
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

def answer_text(res):
    if isinstance(res, dict):
        return str(res.get("text", "") or "")
    if isinstance(res, (list, tuple)):
        return answer_text(res[0]) if res else ""
    return str(res or "")
"""

RUNNER_SRC += r"""
# --- Feed machinery -----------------------------------------------------------------
import csv, glob, hashlib, json, math, re, subprocess
import numpy as np
import librosa
import soundfile as sf
from PIL import Image

CLIPS = "/content/clips"
STILL = os.path.join(CLIPS, "codec_still.png")
TRACK_DIRS = ["/content/drive/MyDrive/codec-control/codec_extracted_tracks",
              "/content/codec_extracted_tracks"]
for d in TRACK_DIRS:
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass

FEED_AAC = ("omni chat units: 1 frame per second + the matching 1s "
            "of the 16 kHz audio track per unit, frames downscaled "
            "to fit 448x448, max 15 units, max_slice_nums=1, "
            "greedy (sampling=False), no generated speech; the "
            "audio track was decoded out of the Arm A MP4 by "
            "ffmpeg and loaded at 16 kHz, the Series Two path")
FEED_RAW = ("omni chat units: the swap still (codec_still.png) + "
            "the matching 1s of the 16 kHz waveform loaded "
            "directly from the WAV file with librosa per unit, "
            "frames downscaled to fit 448x448, max 15 units, "
            "max_slice_nums=1, greedy (sampling=False), no "
            "generated speech; the audio array never passes "
            "through an MP4 or an AAC decode")

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

def waveform_sha256(audio_np):
    arr = np.ascontiguousarray(np.asarray(audio_np, dtype=np.float32))
    return hashlib.sha256(arr.tobytes()).hexdigest()

FRAME_DIR = "/content/codec_frames"
os.makedirs(FRAME_DIR, exist_ok=True)
_feed_cache = {}

def _proof_and_units(path, audio_np, sr, with_picture, frames,
                     frame_rc, audio_rc, feed_text):
    # Shared measurement and assertion core: Arm A calls it with
    # the track extracted from the MP4, Arm B with the waveform
    # loaded from the WAV. The checks are the Series Two ones,
    # plus the battery's duration assertion.
    src_dur = media_duration(path)
    if src_dur is None:
        raise FeedProofError(f"could not read media duration for {path}")
    if abs(src_dur - 10.0) > 0.5:
        raise FeedProofError(
            f"battery file duration is not 10.0 s for {path}: "
            f"{src_dur:.3f}s")
    unit_count = int(round(src_dur))
    if unit_count <= 0:
        raise FeedProofError(f"media duration rounds to zero units: {path}")
    if unit_count > 15:
        unit_count = 15
    track_dur = len(audio_np) / sr if sr else 0.0
    if len(audio_np) == 0:
        raise FeedProofError(f"track is empty for {path}")
    if track_dur < unit_count - 0.5:
        raise FeedProofError(
            f"track is short for {path}: {track_dur:.3f}s of "
            f"track for {unit_count} units")
    if with_picture and frames and len(frames) < unit_count - 1:
        raise FeedProofError(
            f"frame extraction is short for {path}: {len(frames)} "
            f"frames for {unit_count} units")

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
    return {"src_dur": src_dur, "unit_count": unit_count,
            "track_dur": track_dur, "per_unit_rms": per_unit_rms,
            "per_unit_padding": per_unit_padding,
            "zero_chunks": zero_chunks, "overall_rms": overall_rms,
            "nonzero_share": nonzero_share}

def _finish_feed(path, audio_np, sr, m, frame_rc, audio_rc,
                 expected_sound, expected_silence, feed_text,
                 contents):
    if expected_sound and (m["nonzero_share"] == 0.0
                           or m["overall_rms"] <= -119.5
                           or m["zero_chunks"] == m["unit_count"]):
        raise FeedProofError(
            f"track that should carry sound decoded silent for {path}: "
            f"rms={m['overall_rms']} dB, "
            f"nonzero share={m['nonzero_share']}")
    if expected_silence and m["nonzero_share"] > 0.001:
        raise FeedProofError(
            f"declared silence track is not silent for {path}: "
            f"nonzero share={m['nonzero_share']}")
    track_hash = waveform_sha256(audio_np)
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
    proof = {
        "file": os.path.basename(path),
        "source_file_sha256": file_sha256(path),
        "frame_ffmpeg_rc": frame_rc,
        "audio_ffmpeg_rc": audio_rc,
        "track_duration_s": round(m["track_dur"], 3),
        "sample_rate": int(sr),
        "sample_count": int(len(audio_np)),
        "track_rms_db": m["overall_rms"],
        "track_nonzero_share": round(m["nonzero_share"], 6),
        "unit_count": int(m["unit_count"]),
        "per_unit_rms_db": m["per_unit_rms"],
        "per_unit_padding_samples": m["per_unit_padding"],
        "padding_samples_total": int(sum(m["per_unit_padding"])),
        "zero_chunk_count": int(m["zero_chunks"]),
        "track_sha256": track_hash,
        "track_copy_paths": track_paths,
        "expected_sound": bool(expected_sound),
        "expected_silence": bool(expected_silence),
        "feed": feed_text,
    }
    return {"contents": contents, "proof": proof}

def extract_feed(path, expected_sound, expected_silence):
    # Arm A: the Series Two with-picture extraction, unchanged.
    # Frames at 1 fps from the MP4; the soundtrack decoded out
    # of the MP4 to a 16 kHz WAV and loaded with librosa; units
    # of frame plus the matching second of sound.
    key = (path, True, bool(expected_sound), bool(expected_silence))
    if key in _feed_cache:
        return _feed_cache[key]
    if not os.path.exists(path):
        raise FeedProofError(f"feed file missing: {path}")
    tag = re.sub(r"[^A-Za-z0-9]+", "_",
                 os.path.basename(path) + "_aac")
    fdir = os.path.join(FRAME_DIR, tag)
    os.makedirs(fdir, exist_ok=True)
    for old in glob.glob(os.path.join(fdir, "*")):
        os.remove(old)

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

    m = _proof_and_units(path, audio_np, sr, True, frames,
                         frame_rc, audio_rc, FEED_AAC)
    contents = []
    for i in range(m["unit_count"]):
        chunk = audio_np[sr * i: sr * (i + 1)]
        if len(chunk) < sr:
            chunk = np.pad(chunk, (0, sr - len(chunk))).astype(np.float32)
        img = Image.open(frames[min(i, len(frames) - 1)]).convert("RGB")
        img.thumbnail((448, 448))
        contents.extend(["<unit>", img, chunk])
    feed = _finish_feed(path, audio_np, sr, m, frame_rc, audio_rc,
                        expected_sound, expected_silence, FEED_AAC,
                        contents)
    _feed_cache[key] = feed
    return feed

def extract_feed_raw(path, expected_sound, expected_silence):
    # Arm B: the waveform is loaded directly from the WAV file
    # with librosa at 16 kHz mono. No MP4 is opened and no
    # ffmpeg call touches the audio. The units carry the swap
    # still (codec_still.png) and the matching second of that
    # waveform, in the same shape as Arm A's units.
    key = (path, False, bool(expected_sound), bool(expected_silence))
    if key in _feed_cache:
        return _feed_cache[key]
    if not os.path.exists(path):
        raise FeedProofError(f"feed file missing: {path}")
    if not os.path.exists(STILL):
        raise FeedProofError(f"the still image is missing: {STILL}")
    audio_np, sr = librosa.load(path, sr=16000, mono=True)
    audio_np = np.asarray(audio_np, dtype=np.float32)
    frame_rc = ("n/a (the picture is the still image file "
                "codec_still.png, opened directly; this path "
                "makes no ffmpeg call)")
    audio_rc = ("n/a (the waveform is loaded from the WAV "
                "file with librosa; this path makes no ffmpeg "
                "call and opens no MP4)")
    # frames=None: the picture is one still reused for every
    # unit, so the frame-count check (which guards Arm A's
    # per-second frame extraction) does not apply here.
    m = _proof_and_units(path, audio_np, sr, True, None,
                         frame_rc, audio_rc, FEED_RAW)
    contents = []
    for i in range(m["unit_count"]):
        chunk = audio_np[sr * i: sr * (i + 1)]
        if len(chunk) < sr:
            chunk = np.pad(chunk, (0, sr - len(chunk))).astype(np.float32)
        img = Image.open(STILL).convert("RGB")
        img.thumbnail((448, 448))
        contents.extend(["<unit>", img, chunk])
    feed = _finish_feed(path, audio_np, sr, m, frame_rc, audio_rc,
                        expected_sound, expected_silence, FEED_RAW,
                        contents)
    _feed_cache[key] = feed
    return feed

def ask(job, prompt, max_new_tokens):
    path = os.path.join(CLIPS, job["file"])
    if job["arm"] == "raw_wav":
        feed = extract_feed_raw(path, job["expected_sound"],
                                job["expected_silence"])
    else:
        feed = extract_feed(path, job["expected_sound"],
                            job["expected_silence"])
    proof = feed["proof"]
    print("FEED PROOF " + json.dumps(proof, sort_keys=True))
    contents = list(feed["contents"]) + [prompt]
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
"""

RUNNER_SRC += r"""
# --- The battery -------------------------------------------------------------------
# The job list was built and counted by the notebook's builder
# (7 sounds x 2 arms x 2 forms x 2 repeats = 56 asks) and is
# injected here as a literal. The open prompt is the swap
# study's P3, pulled from pilot2/swap/run_swap.py at build
# time; the yes/no prompt is the Series Two template with the
# codec targets.
JOBS = __JOBS__

print(f"codec battery: {len(JOBS)} asks "
      f"(7 sounds x 2 arms x 2 forms x 2 repeats)")

FIELDS = ["test", "ask_id", "sound", "file", "arm", "version",
          "question_form", "question_label", "target", "prompt",
          "repeat", "target_start_s", "expected_sound",
          "expected_silence", "frame_ffmpeg_rc", "audio_ffmpeg_rc",
          "track_duration_s", "sample_rate", "sample_count",
          "track_rms_db", "track_nonzero_share", "unit_count",
          "per_unit_rms_db", "per_unit_padding_samples",
          "padding_samples_total", "zero_chunk_count",
          "track_sha256", "track_copy_paths", "source_file_sha256",
          "raw_answer", "yes_score", "yes_score_status", "model",
          "feed"]

OUT_PATHS = ["/content/drive/MyDrive/codec-control/codec_minicpm_results.csv",
             "/content/codec_minicpm_results.csv"]

def ask_id_of(job, form, target, rep):
    return (f"{job['sound']}|{job['arm']}|{form}|{target}|{rep}")

# --- Battery file assertions: the only gates ---------------------------------------
# Before any ask, every battery file is decoded through the
# very path its arm will ride and asserted. For Arm B the
# waveform handed toward the model must also hash equal to
# an independent direct decode of the same WAV in this same
# session. A failure here stops the run with exit code 7
# before a single question is asked. There is no
# model-answer sanity gate in this runner: a past gate
# misfired on a missing file, so the gates are the files
# themselves.
def assert_battery_files():
    seen = {}
    for job, form, label, target, prompt, rep, max_tokens in JOBS:
        key = (job["file"], job["arm"])
        if key in seen:
            continue
        seen[key] = True
        path = os.path.join(CLIPS, job["file"])
        print("-" * 70)
        print(f"FILE ASSERT {job['arm']}: {job['file']}")
        try:
            if job["arm"] == "raw_wav":
                feed = extract_feed_raw(path, job["expected_sound"],
                                        job["expected_silence"])
                proof = dict(feed["proof"])
                canon, csr = librosa.load(path, sr=16000, mono=True)
                canon_hash = waveform_sha256(canon)
                if canon_hash != proof["track_sha256"]:
                    raise FeedProofError(
                        f"Arm B waveform hash mismatch for {path}: "
                        f"the feed carries {proof['track_sha256']}, "
                        f"a direct decode of the WAV gives "
                        f"{canon_hash}")
                proof["wav_canonical_sha256"] = canon_hash
                native, nsr = sf.read(path, dtype="float32",
                                      always_2d=True)
                native_mono = native.mean(axis=1)
                proof["wav_native_sample_rate"] = int(nsr)
                proof["wav_native_rms_db"] = round(
                    rms_db(native_mono), 2)
                proof["wav_native_duration_s"] = round(
                    len(native_mono) / nsr, 3)
            else:
                feed = extract_feed(path, job["expected_sound"],
                                    job["expected_silence"])
                proof = feed["proof"]
        except FeedProofError as e:
            print()
            print("!" * 70)
            print("BATTERY FILE ASSERTION FAILED:")
            print(str(e))
            print("No ask has been taken. Fix the file set and run")
            print("again. Show this whole output to Muse.")
            print("!" * 70)
            sys.exit(7)
        print("FILE PROOF " + json.dumps(proof, sort_keys=True))
    print()
    print(f"file assertions passed for {len(seen)} files "
          f"(7 sounds x 2 arms).")

assert_battery_files()

if MODE == "check":
    print()
    print("check mode done: proof, ladder, and file assertions")
    print("all passed. The battery was not run. Run Step 4 for the")
    print("56 asks.")
    sys.exit(0)

# --- Resume: load any rows already saved --------------------------------------------
rows = []
done_keys = set()
for p in OUT_PATHS:
    if os.path.exists(p):
        try:
            with open(p, newline="") as fh:
                rows = list(csv.DictReader(fh))
            print(f"resume: loaded {len(rows)} existing rows from {p}")
            break
        except Exception as e:
            print("resume: could not read", p, "-", str(e)[:120])
            rows = []
for r in rows:
    done_keys.add((r.get("sound", ""), r.get("arm", ""),
                   r.get("question_form", ""), r.get("target", ""),
                   str(r.get("repeat", ""))))
failed = 0

def save_rows():
    for out_path in OUT_PATHS:
        try:
            with open(out_path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(rows)
            print("saved:", out_path)
        except Exception as e:
            print("could not save", out_path, "-", str(e)[:120])

for job, form, label, target, prompt, rep, max_tokens in JOBS:
    key = (job["sound"], job["arm"], form, target, str(rep))
    aid = ask_id_of(job, form, target, rep)
    if key in done_keys:
        print("=" * 70)
        print(f"SKIP (already in the CSV): {aid}")
        continue
    print("=" * 70)
    print(f"codec_control - sound={job['sound']} arm={job['arm']} "
          f"version={job['version']} form={form}/{label} "
          f"target={target} repeat={rep}")
    raw_text = ""
    error = ""
    proof = None
    try:
        raw, proof = ask(job, prompt, max_tokens)
        print(f"ASK {aid} arm={job['arm']} file={job['file']} "
              f"waveform_rms_db={proof['track_rms_db']}")
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
        "test": job["test"], "ask_id": aid, "sound": job["sound"],
        "file": job["file"], "arm": job["arm"],
        "version": job["version"], "question_form": form,
        "question_label": label, "target": target, "prompt": prompt,
        "repeat": rep, "target_start_s": job.get("target_start_s", ""),
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

RUNNER_SRC = RUNNER_SRC.replace("__JOBS__", repr(JOBS))
assert "__JOBS__" not in RUNNER_SRC, "job placeholder survived"

# ---------------------------------------------------------------- step 3
cells.append(md(r"""
## Step 3 · Write the runner, then check: ladder and file assertions

The first cell below writes the runner script,
`/content/minicpm_codec_run.py`. It holds the same proof and
the same loading ladder as the Series Two MiniCPM runner
(bf16, then 8-bit, then 4-bit, all pinned to revision
`06849bfd36da94da1f5a88fa17e8ba63f08a4c4d`, remote-code
cache cleared first), with rung tracebacks appended to
`codec_minicpm_load_log.txt` in the clips folder. What
changed is the battery and its gates.

The second cell runs the script in **check mode**: proof
lines (including the yes-score fact: `model.chat` exposes
no score, settled from the code), the ladder, and then the
battery file assertions. Every one of the fourteen battery
files is decoded through the very path its arm will ride,
printed with its SHA-256, duration, and per-second RMS, and
asserted: 10.0 seconds within half a second; sound where
sound is declared, true zeros where silence is declared;
and for the seven WAVs, the waveform handed toward the
model must hash equal to a direct decode of the same WAV.
Any failure stops the run here, exit code 7, before a
single ask.

There is deliberately no sanity question. The gates are the
files.
"""))

cells.append(code(
    "# --- Write the codec control runner script --------------------------------------\n"
    "# The runner holds the ladder, the battery file assertions,\n"
    "# the strict feed proof, and the 56 codec asks. It is\n"
    "# written to a plain file here and only ever run under the\n"
    "# private environment's Python.\n"
    "RUNNER_SOURCE = " + repr(RUNNER_SRC) + "\n"
    "with open(\"/content/minicpm_codec_run.py\", \"w\") as fh:\n"
    "    fh.write(RUNNER_SOURCE)\n"
    "print(\"runner written:\", \"/content/minicpm_codec_run.py\", "
    "len(RUNNER_SOURCE), \"chars\")\n"))

cells.append(code(r"""
# --- Check mode: proof, ladder, file assertions -------------------------------------
import os, subprocess
VENV_PY = "/content/minicpm-venv/bin/python"
proc = subprocess.run([VENV_PY, "/content/minicpm_codec_run.py", "check"])
print()
print("check exit code:", proc.returncode)
if proc.returncode != 0:
    print("!" * 72)
    print("STOP. THE CHECK RUN FAILED. The runner's own output above says")
    print("where: a proof line, a ladder rung, or a battery file")
    print("assertion. Do not run Step 4. Show this output to Muse.")
    print("!" * 72)
    raise SystemExit("STOP: runner check failed")
print("check passed: proof lines right, model loaded, every battery")
print("file asserted on its real feed path. Go on to Step 4.")
"""))

# ---------------------------------------------------------------- step 4
cells.append(md(r"""
## Step 4 · Run the codec battery

The next cell runs the runner in **battery mode**: proof,
ladder, and the file assertions again, then the 56 asks.
Every ask prints its FEED PROOF line before the model's
answer is taken, and an ASK line naming the ask, its arm,
its file, and the RMS of the waveform actually passed.

**How the arms differ, stated plainly.** Arm A rides the
Series Two extraction: frames at one per second out of the
MP4, its soundtrack decoded to 16 kHz, one-second units of
frame plus sound chunk through the omni chat. Arm B loads
the waveform straight from the WAV at 16 kHz and builds the
same units, each carrying the swap still and the matching
second of that waveform. The picture is the same still in
both arms. The sound content is the same sound in both
arms. The only difference is what the sound passed through
on its way in: an AAC encode and decode, or nothing.

**Token caps.** 300 for the open question, 16 for yes/no.
Decoding is greedy everywhere (sampling=False).

**Hard stops and warnings.** A feed that fails its proof
stops the whole run on the spot (exit code 7), with the
rows completed so far saved first. An ask that errors or
comes back empty is saved with its full error text in
`raw_answer` and a WARNING on screen, never as silence,
and the run ends with the RUN PROBLEM banner (exit code
6). If the session dies, run Step 4 again: asks already
in `codec_minicpm_results.csv` are skipped.

Results save to `codec_minicpm_results.csv` in the Drive
clips folder, one row per ask, with the full feed-proof
numbers on every row, in the Series Two schema plus an
ask_id column. The yes-score column reads "not exposed" on
every row, as in this model's Series Two run.
"""))

cells.append(code(r"""
# --- Battery mode: the codec control asks --------------------------------------------
import os, subprocess
VENV_PY = "/content/minicpm-venv/bin/python"
proc = subprocess.run([VENV_PY, "/content/minicpm_codec_run.py", "battery"])
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

Read the two arms against each other, sound by sound,
before reading sounds against each other. The reading was
fixed in advance:

- **Caught raw, missed in AAC:** the file did it. The
  codec, not the model, lost that sound, and any earlier
  Project K miss on a sound like it has to be re-labeled.
- **Missed in both:** the model did it. The sound arrived
  intact (the Arm B proof says so, hash and all) and was
  still not reported. The codec objection dies for that
  sound.
- **Caught in both:** codec loss does not explain that
  sound.
- **Caught in AAC, missed raw:** a build bug, by definition
  of the design. Stop and inspect the feed paths before
  believing any row.

Then compare with this model's Series Two record. In Series
Two, MiniCPM transcribed the speech perfectly through
another door and still reported hearing nothing; the swap
speech here is that same soundtrack, so its two arms speak
directly to whether any part of that reporting loss was
the file's. The silence rows matter in the other direction:
a yes on silence in either arm is a false alarm, and the
arms should agree on zeros, because both proofs verify the
track really is zeros.

- **Scope.** Every claim stays capped at this model,
  these sounds, these two feed paths, this date. Arm B
  proves what arrives when nothing is compressed away; it
  does not prove what any deployed system hears, because
  deployed systems ride files, which is Arm A's world, and
  that is exactly why both arms are in the record.

When you finish, the file `codec_minicpm_results.csv` in
your Drive is the record. Send it back, and the codec
control gets written up next to the Series Two runs.
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

out = os.path.join(HERE, "project_k_minicpm_codec_colab.ipynb")
with open(out, "w") as fh:
    json.dump(nb, fh, indent=1)
print("wrote", out, "cells:", len(cells))
