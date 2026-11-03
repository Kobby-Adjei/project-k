#!/usr/bin/env python3
"""Build project_k_ola_series2_colab.ipynb.

Series Two diagnostic notebook for OLA (7B). It reuses the proven
sealed-venv OLA notebook pattern: Step 1 is taken byte for byte from
the built Series One OLA notebook (same private environment, same
authors' stack, same openai-whisper build-constraints fix, same
installer evidence capture), and Step 3 keeps that notebook's model
download, its config path rewrite, and its loading ladder (bf16,
then the builder's load_8bit, then load_4bit), with full tracebacks
appended live to a load log. What changes is the battery: Step 4
runs the Series Two diagnostic set, the same 87 asks the MiniCPM
Series Two notebook runs (Test 2 the 2x2, 48 asks; Test 3 yes/no
scoring, 30 asks; Test 4 tone-shift timing, 9 asks), on the same
files, with the same question forms, and the same CSV schema,
saving series2_ola_results.csv in the clips folder.

Two OLA feed questions had to be settled from the authors' code
before this build, and both rulings are written into the notebook
itself, in the title cell, the Step 4 notes, and every row's feed
text:

1. The sound-alone arm of Test 2. MiniCPM's runner fed sound alone
   through its unified chat with no picture. OLA's authors ship an
   equivalent door of their own: the audio+text branch of
   inference/infer.py (an audio file given with no image and no
   video; their line 105 sets modality "text"). In that branch the
   question rides as text after the speech token with no image
   token in the prompt (lines 171-172), tokenized by their speech
   tokenizer (line 187), and generate is called with
   modalities=['text'] (line 295) and a blank zero image tensor as
   the vision placeholder. ola/model/ola_arch.py (lines 292-293)
   shows what that placeholder contributes: when the prompt holds
   no image token, the image features are appended as an empty
   [0:0] slice, so no image embedding enters the sequence. The arm
   is therefore implemented through that branch, honestly, with
   the soundtrack taken out of the same swap file by the same
   moviepy step the video door uses. It is a real authors' door,
   not a faked one, and the difference from MiniCPM's sound-alone
   form is stated wherever the results are read.

2. The yes score. MiniCPM's chat door returns decoded text only
   and never requests output scores, so its rows read "not
   exposed". OLA's door is different in the code: the runner (like
   the authors' inference entry) calls model.generate directly,
   and OlaQwenForCausalLM.generate (ola/model/language_model/
   ola_qwen.py, line 168) prepares the multimodal embeddings and
   forwards every remaining keyword to super().generate. Asking
   that same call to also return its per-step scores
   (output_scores with return_dict_in_generate) changes nothing
   about the prompt, the feed, or the greedy decoding; it only
   keeps numbers the call already computes. So on yes/no asks the
   runner records the probability of a yes token at the first
   generated position, read from that call's own first score row.
   No internal path is substituted anywhere: no hidden states, no
   extra forward pass, no hook. Rows that are not yes/no asks
   carry no score, because that question form has no yes token
   to score.

The feed proof follows the audit rule: printed from inside the
session, per ask, before the answer is taken, from OLA's own feed
path. For each ask the runner measures the exact waveform its
loader decodes (duration, sample count, RMS in dB, nonzero share,
and the same numbers per 30 second chunk, which is the chunk unit
OLA's own loader cuts; the MiniCPM CSV's unit columns carry these
chunk numbers here, and the notebook says so), plus, on video
asks, the frame counts (frames in the file, 64 sampled, and how
many the authors' selection check kept). A copy of each extracted
track is saved beside the results and the track is hashed. The
hard stop carries the same thresholds as the MiniCPM Series Two
runner: a sound-carrying track that decodes missing, empty, short
by more than 0.5 s against the file's own duration, or all zeros
stops the run with exit code 7 and the rows so far saved; declared
silence is exempt by name in the job list and is itself verified
to be zeros.

The battery's question strings are not retyped. The repaired JSON
question is pulled from pilot2/run_eggs.py at build time and
asserted at 444 characters, and the whole job list section of the
runner (sounds, prompts, targets, lures, repeats, token caps) is
copied word for word from the MiniCPM Series Two runner, so the
two runs ask byte-identical questions by construction. The build
verification then proves it again by executing both job lists and
comparing them.

Plain language throughout, no em dashes.
"""
import copy
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT2 = os.path.join(HERE, "..", "pilot2")
BASE_NB = os.path.join(HERE, "project_k_ola_colab.ipynb")


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

# The Series One OLA notebook cells are: 0 title, 1 Step 1 markdown,
# 2 Step 1 code. Step 1 is reused byte for byte because the
# environment, the authors' stack, the build-constraints fix, and
# the proof are deliberately identical.
step1_md = copy.deepcopy(base_nb["cells"][1])
step1_code = copy.deepcopy(base_nb["cells"][2])

cells = []
cells.append(md(r"""
# Project K, Series Two: OLA rides the diagnostic cells

**Why this notebook exists.** Series One asked four models the same
questions on the same clips, and each one lost the sound in its own
shape. Series Two asks the next question: which stage loses it? The
council's tests change one thing at a time on the same files. Feed
integrity first (did the sound provably reach the model), then
recognition, interference, reporting, and timing, as hypotheses to
separate, not labels to assume. MiniCPM-o 2.6 ran these cells first
(Nov 2, 2026): its feed was proved per ask, speech was transcribed
word perfect under the direct form and denied under the open and
presence forms, and the vacuum was owned for the first time under a
forced yes or no. OLA runs next, on the same cells, per the run
order set before either model ran.

**The prediction, written down before this run.** Claude's, from
the Series Two plan: if the ear is the bottleneck, OLA should name
the vacuum and the sizzle where the speech-ear-only models did
not. OLA hears through two ears, Whisper large-v3 for speech and
BEATs for sound events; the models that missed the vacuum and the
sizzle hear through a speech ear alone. If OLA also misses them,
the bottleneck is probably not the ear. Either way, the prediction
was registered before the run, and the run does not get to rewrite
it.

**What this notebook does.** It runs OLA (7B) through the same 87
diagnostic asks MiniCPM faced:

1. **Test 2, the 2x2 (48 asks).** Four sounds (speech, true
   silence, water, beeps), each fed as sound alone and as sound
   with the swap still picture, crossed with the open question and
   two direct questions, two repeats per cell.
2. **Test 3, yes/no scoring (30 asks).** The full seven-file swap
   set, in three versions each: sound on with the still picture,
   muted with the still picture, and sound on with a plain gray
   frame.
3. **Test 4, tone-shift timing (9 asks).** One 1000 Hz tone,
   0.5 seconds long, placed at about 2, 6, and 10 seconds in
   12 second files. Presence, early-or-late, and the numeric start
   are asked separately.

Results save as `series2_ola_results.csv` in the Drive clips
folder, in the same schema as MiniCPM's Series Two results, so
the two runs read side by side, row for row.

**The doors, stated up front.** OLA takes a video file per ask
through its authors' inference path, and the two arms of Test 2
ride two branches of that one path:

- **With the picture:** the authors' video branch, exactly as the
  Series One OLA notebook ran it. Moviepy takes the sound out of
  the video itself, decord samples 64 frames evenly over the whole
  clip, the authors' frame selection check is kept as shipped and
  its kept count is recorded per ask, and the question rides after
  the speech token and the image token.
- **Sound alone:** the authors' own audio+text branch of the same
  inference script, the one taken when an audio file is given with
  no image and no video. The soundtrack is taken out of the same
  swap file by the same moviepy step, the question rides as text
  after the speech token, and the prompt holds no image token.
  That branch passes a blank zero image as a placeholder, and the
  authors' own input builder appends that placeholder's features
  as an empty slice when the prompt holds no image token, so no
  picture reaches the sequence. The difference from MiniCPM's
  sound-alone form (its unified chat with no picture) is real and
  is stated here and in every row's feed text: each model's
  sound-alone arm rides that model's own documented door.

**The feed proof rule.** Before every answer is taken, the runner
prints the proof for that exact feed, from inside OLA's own path:
the extracted track's duration, sample rate, sample count, RMS in
dB, nonzero share, the numbers per 30 second chunk (the chunk is
OLA's own unit; its loader cuts sound into 30 second chunks of
480,000 samples, and the CSV's unit columns carry chunk numbers
in this run), the padding each chunk needed, a hash of the exact
decoded track, and, on video asks, the frame counts (in the file,
sampled, kept). A copy of each unique extracted track is saved
beside the results. If a track that should carry sound decodes
missing, empty, short by more than 0.5 seconds against the file's
own duration, or all zeros, the run hard-stops with a loud banner
and the rows so far are saved. Declared silence and declared
muted versions are exempt by name in the job list, never by
accident, and declared silence is itself verified to be zeros.

**The yes-score fact, settled from the code before any run.**
MiniCPM's chat door returns decoded text only and never requests
output scores, so its rows read "not exposed" and its words are
the record. OLA's door is built differently: the runner, like the
authors' inference entry, calls `model.generate` directly, and
the model's own generate method forwards every extra keyword to
the standard Hugging Face generate. Asking that same call to also
return its per-step scores changes nothing about the prompt, the
feed, or the greedy decoding; it only keeps numbers the call
already computes. So on yes/no asks this runner records the
probability of a yes token at the first generated position, read
from that call's own first score row. No internal path is
substituted: no hidden states, no extra forward pass, no hooks.
Rows that are not yes/no asks carry no score, because that
question form has no yes token to score.

**Where it runs.** The paid Colab account (kobbytti) and an
**L4 GPU** with the high-memory runtime, the same as the other
OLA notebook: the weights in full form plus the two speech
encoders want roughly 19 to 20 GB of the L4's 24.

**How to run it.** Run the cells from the top, in order. Step 1
builds the same private environment as the Series One OLA
notebook and proves the stack. Step 2 gets the twelve clips and
the Series Two test files. Step 3 writes the runner and runs its
check mode (proof, download, loading ladder, sanity). Step 4 runs
the diagnostic battery and saves `series2_ola_results.csv`. No
restart is needed anywhere in this notebook.

**The one rule.** We claim only what we ran. Every result here is
about these clips, this model, this feed, this run. Nothing
wider.
"""))
cells.append(step1_md)
cells.append(step1_code)

# ---------------------------------------------------------------- step 2
cells.append(md(r"""
## Step 2 · Get the clips and the Series Two test files

Everything this run asks about already lives in your Google Drive,
in the folder named `project-k-clips`. The next cell links your
Drive, finds that folder, and copies what it needs to a plain work
folder here. Twenty-five files in all:

**The twelve clips** (the same twelve the other notebooks use; the
sanity gate in Step 3 rides one of them):

1. `P02_07_5.mp4`: the vacuum clip.
2. `P32_01_54.mp4`: the jar clip.
3. `own-01.mp4`: your eggs recording.
4. `probe_speech.mp4`: the spoken sentence control.
5. `probe_tone.mp4`: three loud beeps; the sanity gate's clip.
6. `swap_speech.mp4`, 7. `swap_beeps.mp4`, 8. `swap_water.mp4`,
   9. `swap_sizzle.mp4`: four swap cells.
10. `matched_speech.mp4`, 11. `matched_speech_beeps.mp4`,
    12. `matched_speech_vacuum.mp4`: the matched files.

**The rest of the swap set** (one still picture, seven soundtracks,
10 seconds each, loudness matched at build time to about
RMS -20 dB, silence true digital silence):

13. `swap_silence.mp4`, 14. `swap_vacuum.mp4`, 15. `swap_mixed.mp4`.

**The Series Two test files**, built by the MiniCPM Series Two run
and saved into this same Drive folder, so this run rides identical
files, not rebuilt cousins:

16. to 22. **Gray-frame versions** (`series2_gray_<sound>.mp4`):
    each swap soundtrack over a plain gray frame, same size and
    length as the swap still versions.
23. to 25. **Tone files** (`series2_tone_2s.mp4`,
    `series2_tone_6s.mp4`, `series2_tone_10s.mp4`): the swap still
    picture for 12 seconds, carrying one 1000 Hz tone, 0.5 seconds
    long, placed at about 2, 6, and 10 seconds, each file's total
    track level matched to RMS -20 dB.

If a gray or tone file is missing from the Drive folder, the cell
rebuilds just that file here, with the same algorithm the MiniCPM
Series Two notebook used (same gray, same frame size, same tone,
same measured gain loop, same decoded measurement printed), and
saves the rebuild into the Drive clips folder as well. A rebuild
is the fallback, not the plan: the first choice is always the
identical file the MiniCPM run already measured.

The cell also checks each of the twelve clips the way the other
notebooks do: any clip longer than 20 seconds gets cut to its
first 15 seconds (that is the eggs recording). The swap, gray, and
tone files are exact test files and are never cut.

Do not go on until all twenty-five files say "ready".

One thing this cell does not do: check the notebook session's
transformers version. The model runs under the private
environment's Python, whose versions Step 1 proved and the runner
re-proves on every run. This session's versions are simply not
part of the experiment.
"""))

cells.append(code(r"""
# --- Get the clips and the Series Two test files -------------------------------
# No version gate here, on purpose. The model runs under the private
# environment's Python (Step 1), which proves its own versions
# before anything loads. This cell only moves and, if needed,
# builds files. The gray and tone builds below are the MiniCPM
# Series Two notebook's Step 2 algorithm, unchanged, used only for
# files the Drive folder does not already hold.
import os, shutil, subprocess, json, math, wave
import numpy as np

BASE12 = ["P02_07_5.mp4", "P32_01_54.mp4", "own-01.mp4",
          "probe_speech.mp4", "probe_tone.mp4",
          "swap_speech.mp4", "swap_beeps.mp4", "swap_water.mp4",
          "swap_sizzle.mp4",
          "matched_speech.mp4", "matched_speech_beeps.mp4",
          "matched_speech_vacuum.mp4"]
SWAP = ["silence", "speech", "beeps", "water", "vacuum", "sizzle",
        "mixed"]
EXTRA_SWAP = ["swap_silence.mp4", "swap_vacuum.mp4", "swap_mixed.mp4"]
GRAY = [f"series2_gray_{s}.mp4" for s in SWAP]
TONE = [f"series2_tone_{s}s.mp4" for s in (2, 6, 10)]
WORK = "/content/clips"
DRIVE_CLIPS = "/content/drive/MyDrive/project-k-clips"
os.makedirs(WORK, exist_ok=True)

try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception as e:
    print("Drive mount skipped:", e)

SOURCES = [DRIVE_CLIPS, "/content/project-k-clips"]

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

for name in BASE12 + EXTRA_SWAP:
    bring(name)

def duration_s(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path],
        capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except Exception:
        return None

# The twelve clips follow the standing rule of the other
# notebooks: longer than 20 seconds means cut to the first 15.
for name in BASE12:
    path = os.path.join(WORK, name)
    if not os.path.exists(path):
        continue
    d = duration_s(path)
    if d is not None and d > 20:
        short = os.path.join(WORK, "short_" + name)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                        "-t", "15", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", short],
                       check=False)
        if os.path.exists(short):
            os.replace(short, path)
            print("cut", name, "to its first 15 seconds")

# The Series Two test files: take the identical files the MiniCPM
# Series Two run built into the Drive clips folder, when they are
# there. Build only what is still missing.
for name in GRAY + TONE:
    bring(name)

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
    # Keep the work copy and, when Drive is mounted, the clips
    # folder copy.
    if os.path.isdir(DRIVE_CLIPS):
        shutil.copy(local_path, os.path.join(
            DRIVE_CLIPS, os.path.basename(local_path)))
        print("saved to Drive clips folder:",
              os.path.basename(local_path))

build_log = []

# ---- gray-frame versions, built only where missing -----------------------------
for sound in SWAP:
    out = os.path.join(WORK, f"series2_gray_{sound}.mp4")
    if os.path.exists(out):
        continue
    src = os.path.join(WORK, f"swap_{sound}.mp4")
    if not os.path.exists(src):
        raise SystemExit(f"STOP: cannot build the gray version; "
                         f"swap_{sound}.mp4 is missing")
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
             "nonzero_share": round(float(np.count_nonzero(x) / len(x)), 4),
             "built_by": "ola series2 step 2 fallback (minicpm algorithm)"}
    build_log.append(entry)
    save_both(out)
    print("gray (built):", entry)

# ---- tone files, built only where missing --------------------------------------
missing_tones = [s for s in (2, 6, 10)
                 if not os.path.exists(
                     os.path.join(WORK, f"series2_tone_{s}s.mp4"))]
if missing_tones:
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
                "measured_onset_s": (None if onset is None
                                     else round(onset, 3)),
                "nonzero_share": round(
                    float(np.count_nonzero(x) / len(x)), 4)}

    for start_s in missing_tones:
        out = os.path.join(WORK, f"series2_tone_{start_s}s.mp4")
        wav_path = os.path.join(WORK, f"series2_tone_{start_s}s.wav")
        amplitude = 0.6928
        entry = None
        for attempt in range(6):
            write_tone_wav(wav_path, start_s, amplitude)
            proc = subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-loop", "1",
                 "-framerate", "25", "-i", still, "-i", wav_path,
                 "-t", "12", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 "-r", "25", "-c:a", "aac", "-ar", str(TONE_SR),
                 "-shortest", out])
            if proc.returncode != 0 or not os.path.exists(out):
                raise SystemExit(f"STOP: tone build failed for {start_s}s")
            entry = measure_tone(out)
            if abs(entry["audio_rms_db_decoded"] - TARGET_RMS) <= 0.7:
                break
            amplitude *= 10 ** ((TARGET_RMS
                                 - entry["audio_rms_db_decoded"]) / 20.0)
            if amplitude >= 0.98:
                raise SystemExit(
                    f"STOP: tone gain loop ran away for {start_s}s")
        entry = {"file": os.path.basename(out),
                 "target_start_s": start_s, **entry,
                 "amplitude": round(float(amplitude), 4),
                 "built_by": "ola series2 step 2 fallback (minicpm algorithm)"}
        build_log.append(entry)
        save_both(out)
        print("tone (built):", entry)

if build_log:
    log_path = os.path.join(WORK, "series2_ola_build_log.json")
    with open(log_path, "w") as fh:
        json.dump(build_log, fh, indent=2)
    if os.path.isdir(DRIVE_CLIPS):
        shutil.copy(log_path, os.path.join(
            DRIVE_CLIPS, "series2_ola_build_log.json"))
    print("fallback builds logged:", log_path)

print()
ready = True
for name in BASE12 + EXTRA_SWAP + GRAY + TONE:
    path = os.path.join(WORK, name)
    if os.path.exists(path):
        print("ready:  ", name, f"({os.path.getsize(path) / 1e6:.1f} MB)")
    else:
        ready = False
        print("MISSING:", name)
if not ready:
    print()
    print("!" * 72)
    print("STOP. Some files are missing. Check the Drive folder name is")
    print("exactly project-k-clips, or upload the files by hand, then")
    print("run this cell again. Do not go on to Step 3.")
    print("!" * 72)
    raise SystemExit("STOP: files missing in Step 2")
print()
print("ready: all 12 clips, all 7 swap files, 7 gray versions, 3 tone files")
"""))

# ------------------------------------------------------------- runner script
# The runner is a plain Python script the notebook writes to
# /content/ola_series2_run.py and runs under the private
# environment's Python. It holds the model download, the loading
# ladder, the sanity check, the strict feed proof, and the Series
# Two diagnostic battery. The with-picture ask path is the authors'
# video route from inference/infer.py, lifted into a function as
# in the Series One OLA runner. The sound-alone ask path is the
# same script's audio+text branch, for the reasons and code
# evidence set out in this builder's docstring. The battery section
# (job list, prompts, targets, lures) is copied word for word from
# the MiniCPM Series Two runner.
RUNNER_SRC = r"""
# ola_series2_run.py - the Series Two diagnostic runner for OLA (7B).
# Written by project_k_ola_series2_colab.ipynb. Run only under the
# private environment's Python (/content/ola-venv/bin/python).
# Modes: "check" runs the proof, the model download, the loading
# ladder, and the sanity question, then stops. "battery" does all
# of that and then runs the Series Two diagnostic tests and saves
# the CSV.
import os

# The authors' environment variables, set before anything of theirs
# is imported: ola/mm_utils.py and the vision tower read them at
# import time. These are the values inference/infer.py sets, word
# for word.
for _k, _v in {'LOWRES_RESIZE': '384x32', 'HIGHRES_BASE': '0x32',
               'VIDEO_RESIZE': '0x64', 'VIDEO_MAXRES': '480',
               'VIDEO_MINRES': '288', 'MAXRES': '1536', 'MINRES': '0',
               'FORCE_NO_DOWNSAMPLE': '1', 'LOAD_VISION_EARLY': '1',
               'PAD2STRIDE': '1'}.items():
    os.environ[_k] = _v
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else "check"
print("runner mode:", MODE)
print("runner python:", sys.version.split()[0], sys.executable)

# --- Proof lines, printed before anything loads --------------------------------
# The same proof Step 1 ran, repeated here so every run of this
# script carries its own evidence: the versions it actually
# imported, from the paths it actually loaded them, and the imports
# the model's code demands by name.
import torch, transformers
print("torch version:", torch.__version__)
print("transformers version:", transformers.__version__)
print("transformers file:", transformers.__file__)
import flash_attn
from flash_attn import flash_attn_func, flash_attn_varlen_func
print("flash_attn version:", flash_attn.__version__)
print("flash_attn import: OK")
import deepspeed
print("deepspeed version:", deepspeed.__version__)
print("deepspeed import: OK")
from ola.model.builder import load_pretrained_model
import ola.model.builder as _ola_builder
print("ola builder file:", _ola_builder.__file__)
print("ola imports: OK")
if transformers.__version__ != "4.43.4":
    print("STOP: this script is running on transformers",
          transformers.__version__, "and needs exactly 4.43.4, the",
          "version the authors pin and the model config stamps.")
    sys.exit(3)

# The yes-score question, settled from the model's own code before
# the run (ola/model/language_model/ola_qwen.py in the pinned code
# commit): OlaQwenForCausalLM.generate prepares the multimodal
# embeddings itself, then calls super().generate with every
# remaining keyword forwarded. The standard generate therefore
# accepts output_scores and return_dict_in_generate, and those
# flags only add the per-step scores to the same greedy call's
# return; they cannot change the prompt, the feed, or the tokens
# chosen. MiniCPM's chat door, by contrast, returns decoded text
# only, which is why its Series Two rows read "not exposed". Here
# the score is exposed by the door itself, so YES_SCORE_EXPOSED is
# True, and yes/no asks record the probability of a yes token at
# the first generated position from that call's first score row.
YES_SCORE_EXPOSED = True
print("yes token score exposed by the generate door:", YES_SCORE_EXPOSED,
      "(generate is called directly here, as in the authors' inference",
      "entry, and its own scores are returned on request)")

# --- The model files: one pinned revision, downloaded once ----------------------
# The hub repo is downloaded at one exact revision into a plain
# local folder. The config inside it points at the two speech
# encoder files with relative paths (./large-v3.pt and the BEATs
# checkpoint, both of which ship inside the repo). The model card's
# own step 2 says to replace those paths with the local ones; the
# runner does exactly that, in this downloaded copy, by absolute
# path, and prints what it wrote.
import json
from huggingface_hub import snapshot_download

REPO = "THUdyh/Ola-7b"
REVISION = "d2fcaa533792480ce748321bea5101fed29d857e"
MODEL_DIR = "/content/ola-model"

if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
    snapshot_download(repo_id=REPO, revision=REVISION, local_dir=MODEL_DIR)
    print("downloaded:", REPO, "at pinned revision", REVISION)
else:
    print("model dir already present:", MODEL_DIR)
_cfg_path = os.path.join(MODEL_DIR, "config.json")
with open(_cfg_path) as _fh:
    _cfg = json.load(_fh)
_cfg["speech_encoder"] = os.path.join(MODEL_DIR, "large-v3.pt")
_cfg["music_encoder"] = os.path.join(
    MODEL_DIR, "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt")
with open(_cfg_path, "w") as _fh:
    json.dump(_cfg, _fh, indent=2)
print("config speech_encoder:", _cfg["speech_encoder"])
print("config music_encoder:", _cfg["music_encoder"])

# --- The load log ----------------------------------------------------------------
# A failed rung prints its full traceback (never a cut-short
# reason) and the same words are appended, as events happen, to
# series2_ola_load_log.txt, in the Drive clips folder first and
# /content second, so the file is already on disk even if the
# session dies right after a rung does.
import traceback, datetime, gc

LOAD_LOG_PATHS = ["/content/drive/MyDrive/project-k-clips/series2_ola_load_log.txt",
                  "/content/series2_ola_load_log.txt"]

def log_load(text):
    for p in LOAD_LOG_PATHS:
        try:
            with open(p, "a") as fh:
                fh.write(text)
        except Exception:
            pass   # that home is not mounted; the printed copy stands

def log_paths_saved():
    return [p for p in LOAD_LOG_PATHS if os.path.exists(p)]

CODE_COMMIT = "(not recorded)"
try:
    with open("/content/ola-code-commit.txt") as _fh:
        CODE_COMMIT = _fh.read().strip()
except Exception:
    pass

log_load("=" * 70 + "\n")
log_load("OLA Series Two load log - "
         + datetime.datetime.now().isoformat(timespec="seconds")
         + " - runner mode: " + MODE + "\n")
log_load("repo: " + REPO + " at pinned revision " + REVISION + "\n")
log_load("code: github.com/Ola-Omni/Ola at commit " + CODE_COMMIT + "\n")
print("ola code commit for this run:", CODE_COMMIT)

# --- Load the model: the ladder, through the authors' own builder ----------------
# Three rungs, all of them the same model, THUdyh/Ola-7b, loaded by
# the authors' load_pretrained_model, exactly as the Series One
# OLA runner loads it. Rung 1 is full form, 16-bit (bfloat16), the
# authors' default, finished the way their inference script
# finishes it (to cuda, eval, bfloat16). Rungs 2 and 3 are the
# builder's own squeeze flags, load_8bit and load_4bit, each with
# device_map passed through to the loader so the squeezed weights
# land on the chip directly. If every rung fails, the run stops
# here and its closing lines name the log file: no substitute
# model is ever loaded quietly.
def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

tokenizer = None
model = None
image_processor = None
MODEL_NAME = None   # repo name of the model that loaded
LOAD_MODE = None    # the fit that worked: bf16, 8-bit, or 4-bit

def try_load(mode):
    from ola.model.builder import load_pretrained_model
    from ola.utils import disable_torch_init
    disable_torch_init()
    if mode == "bf16":
        tok, m, proc, _ctx = load_pretrained_model(MODEL_DIR, None)
        m = m.to("cuda").eval()
        m = m.bfloat16()
    elif mode == "8-bit":
        tok, m, proc, _ctx = load_pretrained_model(
            MODEL_DIR, None, load_8bit=True, device_map="cuda")
        m = m.eval()
    elif mode == "4-bit":
        tok, m, proc, _ctx = load_pretrained_model(
            MODEL_DIR, None, load_4bit=True, device_map="cuda")
        m = m.eval()
    return tok, m, proc

rung_errors = []
for mode in ("bf16", "8-bit", "4-bit"):
    print("=" * 70)
    print(f"rung: {REPO} in {mode} form")
    try:
        t, m, proc = try_load(mode)
        tokenizer, model, image_processor = t, m, proc
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
        tokenizer = None
        model = None
        image_processor = None
        free_memory()

print()
if model is None:
    print("!" * 70)
    print("ALL THREE RUNGS FAILED. OLA did not load in any fit.")
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
print("environment: private venv /content/ola-venv, python 3.10,",
      "torch 2.1.2+cu121, the authors' pinned stack")
if torch.cuda.is_available():
    used = torch.cuda.memory_allocated() / 1e9
    print(f"chip memory already used by the model: {used:.1f} GB")

# The yes token ids, read from the model's own tokenizer: the
# first token of "Yes" and of " Yes". A yes score is the summed
# probability of those ids at the first generated position.
YES_TOKEN_IDS = []
for _word in ("Yes", " Yes"):
    _enc = tokenizer.encode(_word, add_special_tokens=False)
    if _enc:
        YES_TOKEN_IDS.append(int(_enc[0]))
YES_TOKEN_IDS = sorted(set(YES_TOKEN_IDS))
print("yes token ids for scoring:", YES_TOKEN_IDS,
      [tokenizer.decode([i]) for i in YES_TOKEN_IDS])
"""

RUNNER_SRC += r"""
# --- Feed proof machinery ---------------------------------------------------------
# The audit rule: the proof is printed from inside the session, per
# ask, before the answer is taken, and it measures what OLA's own
# path actually extracted. The audio numbers describe the exact
# waveform the authors' loader decodes: the moviepy-written wav,
# loaded at 16 kHz with the same call the loader itself makes,
# before its 30 second chunking pads anything. The chunk is OLA's
# own unit (480000 samples, 30 s at 16 kHz), so the CSV's unit
# columns carry per-chunk numbers in this run: each chunk's RMS
# over its real samples, the padding that chunk needs to reach
# 480000, and how many chunks are all zeros. The hard-stop
# thresholds are the MiniCPM Series Two runner's: a track that
# should carry sound decoding missing, empty, short by more than
# 0.5 s against the file's own duration, or all zeros stops the
# run; declared silence is exempt by name and verified to be zeros.
import csv, hashlib, math, re, shutil, subprocess
import numpy as np
import librosa
import whisper
import moviepy.editor as mp
from decord import VideoReader, cpu
from PIL import Image
from ola.conversation import conv_templates, SeparatorStyle
from ola.datasets.preprocess import (tokenizer_speech_image_token,
                                     tokenizer_speech_token)
from ola.mm_utils import KeywordsStoppingCriteria, process_anyres_video
from ola.constants import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                           DEFAULT_SPEECH_TOKEN, SPEECH_TOKEN_INDEX)

CLIPS = "/content/clips"
AUDIO_DIR = "/content/ola_audio"
os.makedirs(AUDIO_DIR, exist_ok=True)
TRACK_DIRS = ["/content/drive/MyDrive/project-k-clips/series2_ola_extracted_tracks",
              "/content/series2_ola_extracted_tracks"]
for d in TRACK_DIRS:
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass

CHUNK_SAMPLES = 480000   # the authors' 30 s chunk at 16 kHz

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

def load_audio(audio_file_name):
    # The authors' loader, as shipped in inference/infer.py: 16 kHz
    # mono, the sound cut into 30 second chunks (480000 samples),
    # each chunk padded or trimmed to length, whisper log-mel with
    # 128 mels per chunk, at most 25 chunks kept. The raw waveform
    # chunks ride along because the BEATs half of the dual encoder
    # reads them.
    speech_wav, samplerate = librosa.load(audio_file_name, sr=16000)
    if len(speech_wav.shape) > 1:
        speech_wav = speech_wav[:, 0]
    speech_wav = speech_wav.astype(np.float32)
    CHUNK_LIM = 480000
    speechs = []
    speech_wavs = []
    if len(speech_wav) <= CHUNK_LIM:
        speech = whisper.pad_or_trim(speech_wav)
        speech_wav = whisper.pad_or_trim(speech_wav)
        speechs.append(speech)
        speech_wavs.append(torch.from_numpy(speech_wav).unsqueeze(0))
    else:
        for i in range(0, len(speech_wav), CHUNK_LIM):
            chunk = speech_wav[i : i + CHUNK_LIM]
            if len(chunk) < CHUNK_LIM:
                chunk = whisper.pad_or_trim(chunk)
            speechs.append(chunk)
            speech_wavs.append(torch.from_numpy(chunk).unsqueeze(0))
    mels = []
    for chunk in speechs:
        chunk = whisper.log_mel_spectrogram(chunk, n_mels=128).permute(1, 0).unsqueeze(0)
        mels.append(chunk)
    mels = torch.cat(mels, dim=0)
    speech_wavs = torch.cat(speech_wavs, dim=0)
    if mels.shape[0] > 25:
        mels = mels[:25]
        speech_wavs = speech_wavs[:25]
    speech_length = torch.LongTensor([mels.shape[1]] * mels.shape[0])
    speech_chunks = torch.LongTensor([mels.shape[0]])
    return mels, speech_length, speech_chunks, speech_wavs

def extract_track(path):
    # The extraction both of the authors' doors use: moviepy takes
    # the sound out of the video file itself and writes it as a
    # wav. Any failure here is a feed failure, named as one.
    tag = re.sub(r"[^A-Za-z0-9]+", "_", os.path.basename(path))
    wav_path = os.path.join(AUDIO_DIR, tag + ".wav")
    try:
        clip = mp.VideoFileClip(path)
        try:
            if clip.audio is None:
                raise FeedProofError(
                    "moviepy found no audio track in " + path)
            clip.audio.write_audiofile(wav_path, logger=None)
        finally:
            clip.close()
    except FeedProofError:
        raise
    except Exception as e:
        raise FeedProofError(f"audio extraction failed for {path}: {e}")
    if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
        raise FeedProofError("moviepy wrote no audio file for " + path)
    return wav_path

def prove_track(wav_path, media_path, expected_sound, expected_silence):
    # Measure the waveform the authors' loader will decode, from
    # the wav the extraction just wrote, with the loader's own
    # decode settings (librosa, 16 kHz, mono). Raises
    # FeedProofError on any failed check; returns the proof dict
    # on success. The frame and feed fields are filled in by the
    # ask functions, which know which door is being ridden.
    audio_np, sr = librosa.load(wav_path, sr=16000, mono=True)
    audio_np = np.asarray(audio_np, dtype=np.float32)
    if audio_np.ndim > 1:
        audio_np = audio_np[:, 0]
    src_dur = media_duration(media_path)
    if src_dur is None:
        raise FeedProofError(
            f"could not read media duration for {media_path}")
    track_dur = len(audio_np) / sr if sr else 0.0
    if len(audio_np) == 0:
        raise FeedProofError(f"extracted track is empty for {media_path}")
    if track_dur < src_dur - 0.5:
        raise FeedProofError(
            f"extracted track is short for {media_path}: "
            f"{track_dur:.3f}s of track for a {src_dur:.3f}s file")
    overall_rms = round(rms_db(audio_np), 2)
    nonzero_share = float(np.count_nonzero(audio_np) / len(audio_np))
    n_chunks = int(math.ceil(len(audio_np) / CHUNK_SAMPLES))
    if n_chunks <= 0:
        n_chunks = 1
    per_chunk_rms = []
    per_chunk_padding = []
    zero_chunks = 0
    for i in range(n_chunks):
        seg = audio_np[i * CHUNK_SAMPLES:(i + 1) * CHUNK_SAMPLES]
        per_chunk_rms.append(round(rms_db(seg), 2))
        per_chunk_padding.append(int(CHUNK_SAMPLES - len(seg)))
        if len(seg) and not np.any(seg):
            zero_chunks += 1
    if expected_sound and (nonzero_share == 0.0 or overall_rms <= -119.5
                           or zero_chunks == n_chunks):
        raise FeedProofError(
            f"track that should carry sound decoded silent for "
            f"{media_path}: rms={overall_rms} dB, "
            f"nonzero share={nonzero_share}")
    if expected_silence and nonzero_share > 0.001:
        raise FeedProofError(
            f"declared silence track is not silent for {media_path}: "
            f"nonzero share={nonzero_share}")
    track_hash = hashlib.sha256(audio_np.tobytes()).hexdigest()
    track_name = (re.sub(r"[^A-Za-z0-9]+", "_", os.path.basename(media_path))
                  + "__" + track_hash[:12] + ".wav")
    track_paths = []
    for d in TRACK_DIRS:
        try:
            tp = os.path.join(d, track_name)
            if not os.path.exists(tp):
                shutil.copy(wav_path, tp)
            track_paths.append(tp)
        except Exception:
            pass
    proof = {
        "file": os.path.basename(media_path),
        "source_file_sha256": file_sha256(media_path),
        "frame_ffmpeg_rc": ("n/a (decord reads frames in-process; "
                            "OLA's path makes no ffmpeg call)"),
        "audio_ffmpeg_rc": ("n/a (moviepy extracts in-process; "
                            "OLA's path makes no ffmpeg call)"),
        "media_duration_s": round(src_dur, 3),
        "extracted_wav_path": wav_path,
        "track_duration_s": round(track_dur, 3),
        "sample_rate": int(sr),
        "sample_count": int(len(audio_np)),
        "track_rms_db": overall_rms,
        "track_nonzero_share": round(nonzero_share, 6),
        "unit_count": int(n_chunks),
        "per_unit_rms_db": per_chunk_rms,
        "per_unit_padding_samples": per_chunk_padding,
        "padding_samples_total": int(sum(per_chunk_padding)),
        "zero_chunk_count": int(zero_chunks),
        "track_sha256": track_hash,
        "track_copy_paths": track_paths,
        "expected_sound": bool(expected_sound),
        "expected_silence": bool(expected_silence),
    }
    return proof

# The feeds, stated exactly. The video text is the Series One OLA
# feed note, extended with the file's total frame count and the
# per-ask token cap. The sound-alone text states the authors'
# audio+text door and the placeholder fact, with its code address.
FEED_VIDEO_TEMPLATE = (
    "authors' video path (Ola repo, inference/infer.py): the "
    "clip's own sound is taken out of the video by moviepy and "
    "decoded at 16 kHz; whisper log-mel chunks of 30 s (128 mels) "
    "feed the Whisper large-v3 half of the dual speech encoder "
    "and the raw waveform feeds the BEATs half; decord samples "
    "64 frames evenly over the whole clip and each frame goes "
    "through process_anyres_video at the authors' resize "
    "settings (VIDEO_MAXRES 480, VIDEO_MINRES 288); the authors' "
    "own frame selection check then kept {kept} of the {sampled} "
    "sampled frames (of {total} frames in the file), and that "
    "kept set is the picture the model saw; the question rides "
    "after the speech token and the image token in the qwen_1_5 "
    "template; greedy decoding (temperature 0, sampling off), "
    "at most {tokens} new tokens")
FEED_AUDIO_TEMPLATE = (
    "authors' audio+text path (Ola repo, inference/infer.py, the "
    "branch taken when an audio file is given with no image and "
    "no video; the script names that modality 'text'): the "
    "soundtrack is taken out of the same file by the same "
    "moviepy step the video door uses, then loaded alone; "
    "whisper log-mel chunks of 30 s (128 mels) feed the Whisper "
    "large-v3 half of the dual speech encoder and the raw "
    "waveform feeds the BEATs half; the question rides as text "
    "after the speech token, with no image token in the prompt, "
    "tokenized by the authors' speech tokenizer, and generate "
    "is called with modalities=['text']; the blank zero image "
    "tensor that branch passes as the vision placeholder is "
    "encoded by the vision tower but contributes no tokens to "
    "the sequence, because the prompt holds no image token "
    "(ola/model/ola_arch.py appends its features as an empty "
    "slice in that case); greedy decoding (temperature 0, "
    "sampling off), at most {tokens} new tokens")

def yes_probability(scores):
    # The probability of a yes token at the first generated
    # position, from the first score row of the same generate
    # call that produced the answer. Softmax over the vocabulary,
    # summed over the yes token ids read from the model's own
    # tokenizer at load time.
    if not scores:
        return None
    logits = scores[0][0].float()
    probs = torch.softmax(logits, dim=-1)
    total = 0.0
    for i in YES_TOKEN_IDS:
        total += float(probs[i].item())
    return total

def run_generate(input_ids, media_kwargs, conv, max_new_tokens,
                 want_scores):
    # One generate call through the model's own generate method,
    # greedy, with the authors' stopping criteria, exactly as both
    # of the authors' doors shape it. On yes/no asks the call is
    # also asked to return its per-step scores; that flag adds
    # numbers to the return and changes nothing else.
    attention_masks = input_ids.ne(151643).long().to("cuda")
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer,
                                                 input_ids)
    kwargs = dict(inputs=input_ids,
                  attention_mask=attention_masks,
                  use_cache=True,
                  stopping_criteria=[stopping_criteria],
                  do_sample=False,
                  num_beams=1,
                  max_new_tokens=max_new_tokens)
    kwargs.update(media_kwargs)
    if want_scores:
        kwargs["return_dict_in_generate"] = True
        kwargs["output_scores"] = True
    with torch.inference_mode():
        out = model.generate(**kwargs)
    if want_scores:
        output_ids = out.sequences
        scores = out.scores
    else:
        output_ids = out
        scores = None
    text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    text = text.strip()
    if text.endswith(stop_str):
        text = text[:-len(stop_str)]
    text = text.strip()
    return text, scores

def speech_tensors(wav_path):
    speech, speech_length, speech_chunk, speech_wav = load_audio(wav_path)
    return ([speech.bfloat16().to("cuda")],
            [speech_length.to("cuda")],
            [speech_chunk.to("cuda")],
            [speech_wav.to("cuda")])

def ask_video(path, prompt, max_new_tokens, want_yes,
              expected_sound, expected_silence):
    # One question about one clip, through the authors' video
    # door. The feed proof prints before the answer is taken.
    if not os.path.exists(path):
        raise FeedProofError(f"feed file missing: {path}")
    wav_path = extract_track(path)
    proof = prove_track(wav_path, path, expected_sound, expected_silence)
    speechs, speech_lengths, speech_chunks, speech_wavs = speech_tensors(
        wav_path)
    try:
        vr = VideoReader(path, ctx=cpu(0))
        total_frame_num = len(vr)
        frame_idx = np.linspace(0, total_frame_num - 1, 64,
                                dtype=int).tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
    except FeedProofError:
        raise
    except Exception as e:
        raise FeedProofError(f"frame reading failed for {path}: {e}")
    video = [Image.fromarray(f) for f in spare_frames]
    qs = DEFAULT_SPEECH_TOKEN + DEFAULT_IMAGE_TOKEN + "\n" + prompt
    conv = conv_templates["qwen_1_5"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()
    input_ids = tokenizer_speech_image_token(
        full_prompt, tokenizer, IMAGE_TOKEN_INDEX,
        return_tensors="pt").unsqueeze(0).to("cuda")
    # Frames through the authors' video processing, with their
    # selection check kept exactly as shipped, as in the Series
    # One runner: a sampled frame is kept only if its position in
    # the sampled list also appears in the sampled index list.
    video_processed = []
    for idx, frame in enumerate(video):
        image_processor.do_resize = False
        image_processor.do_center_crop = False
        frame = process_anyres_video(frame, image_processor)
        if frame_idx is not None and idx in frame_idx:
            video_processed.append(frame.unsqueeze(0))
        elif frame_idx is None:
            video_processed.append(frame.unsqueeze(0))
    if not video_processed:
        raise FeedProofError(
            "the authors' frame selection kept no frames for " + path)
    video_processed = torch.cat(video_processed, dim=0).bfloat16().to("cuda")
    kept = int(video_processed.shape[0])
    proof["frames_total"] = int(total_frame_num)
    proof["frames_sampled"] = int(len(video))
    proof["frames_kept"] = kept
    proof["feed"] = FEED_VIDEO_TEMPLATE.format(
        kept=kept, sampled=len(video), total=int(total_frame_num),
        tokens=max_new_tokens)
    print("FEED PROOF " + json.dumps(proof, sort_keys=True))
    text, scores = run_generate(
        input_ids,
        dict(images=video_processed, images_highres=video_processed,
             modalities="video", speech=speechs,
             speech_lengths=speech_lengths, speech_chunks=speech_chunks,
             speech_wav=speech_wavs),
        conv, max_new_tokens, want_yes)
    yp = yes_probability(scores) if want_yes else None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return text, proof, yp

def ask_audio(path, prompt, max_new_tokens, want_yes,
              expected_sound, expected_silence):
    # One question about one soundtrack, through the authors'
    # audio+text door (inference/infer.py, the branch for an
    # audio file with no image and no video). The soundtrack comes
    # out of the same file by the same moviepy step the video door
    # uses; the question rides as text after the speech token; the
    # blank zero image placeholders are passed exactly as that
    # branch passes them. The feed proof prints before the answer
    # is taken.
    if not os.path.exists(path):
        raise FeedProofError(f"feed file missing: {path}")
    wav_path = extract_track(path)
    proof = prove_track(wav_path, path, expected_sound, expected_silence)
    speechs, speech_lengths, speech_chunks, speech_wavs = speech_tensors(
        wav_path)
    blank = torch.zeros(1, 3, 224, 224).to(
        dtype=torch.bfloat16, device="cuda", non_blocking=True)
    blank_highres = torch.zeros(1, 3, 224, 224).to(
        dtype=torch.bfloat16, device="cuda", non_blocking=True)
    images_arg = [blank]
    images_highres_arg = [blank_highres]
    image_sizes = [(224, 224)]
    qs = DEFAULT_SPEECH_TOKEN + "\n" + prompt
    conv = conv_templates["qwen_1_5"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()
    input_ids = tokenizer_speech_token(
        full_prompt, tokenizer, SPEECH_TOKEN_INDEX,
        return_tensors="pt").unsqueeze(0).to("cuda")
    proof["frames_total"] = None
    proof["frames_sampled"] = None
    proof["frames_kept"] = None
    proof["feed"] = FEED_AUDIO_TEMPLATE.format(tokens=max_new_tokens)
    print("FEED PROOF " + json.dumps(proof, sort_keys=True))
    text, scores = run_generate(
        input_ids,
        dict(images=images_arg, images_highres=images_highres_arg,
             image_sizes=image_sizes, modalities=["text"],
             speech=speechs, speech_lengths=speech_lengths,
             speech_chunks=speech_chunks, speech_wav=speech_wavs),
        conv, max_new_tokens, want_yes)
    yp = yes_probability(scores) if want_yes else None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return text, proof, yp

def ask(job, prompt, max_tokens):
    path = os.path.join(CLIPS, job["file"])
    want_yes = prompt.rstrip().endswith("Answer yes or no.")
    if job["arm"] == "audio_alone":
        return ask_audio(path, prompt, max_tokens, want_yes,
                         job["expected_sound"], job["expected_silence"])
    return ask_video(path, prompt, max_tokens, want_yes,
                     job["expected_sound"], job["expected_silence"])

# --- Sanity check: can the model answer at all, through the real door? -----------
# OLA's documented doors all carry media, so this gate rides the
# full video path on the smallest clip in the folder, the beep
# probe, with an instruction whose answer must contain the word
# ready. It exercises the whole chain (decord, moviepy, both
# speech encoders, the vision tower, generate, decode) before the
# battery spends 87 asks on it. A wrong word, an empty answer, or
# an error all stop the run here, loudly. Both modes run it.
SANITY_CLIP = "/content/clips/probe_tone.mp4"
try:
    sraw, _sproof, _syp = ask_video(SANITY_CLIP,
                                    "Reply with the word ready.",
                                    300, False, True, False)
    print("SANITY RAW OUTPUT:", repr(sraw))
    if not sraw:
        print("WARNING: the sanity check came back EMPTY.")
        print("STOP. Show this output to Muse.")
        sys.exit(5)
    if "ready" not in sraw.lower():
        print("WARNING: the sanity answer does not contain the word ready.")
        print("The ask path ran, but the model did not follow the one")
        print("instruction it was given. STOP. Show this output to Muse.")
        sys.exit(5)
    print("sanity check passed: the model answered through the full")
    print("video and sound path, and the word ready is in its answer.")
except SystemExit:
    raise
except Exception as e:
    print("WARNING: the sanity check FAILED with an error:")
    print(str(e))
    print("STOP. Show this output to Muse.")
    sys.exit(5)

if MODE == "check":
    print()
    print("check mode done: proof, download, ladder, and sanity all passed.")
    print("The diagnostic battery was not run. Run Step 4 for the tests.")
    sys.exit(0)
"""

RUNNER_SRC += r"""
# --- Diagnostic battery ---------------------------------------------------------
# The repaired JSON question, injected at build time from
# pilot2/run_eggs.py, character for character (444 characters).
PROMPT_JSON = __PROMPT_JSON__
assert len(PROMPT_JSON) == 444, f"prompt drifted: {len(PROMPT_JSON)} chars"

print("FEED SETTINGS, video asks:", FEED_VIDEO_TEMPLATE.format(
    kept="N", sampled=64, total="T", tokens="M"))
print("FEED SETTINGS, sound-alone asks:", FEED_AUDIO_TEMPLATE.format(
    tokens="M"))
print("MODEL:", MODEL_NAME, "in", LOAD_MODE, "form")

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

YES_STATUS_EXPOSED = ("exposed: the probability of a yes token at the "
                      "first generated position, read from the "
                      "output_scores of the same greedy generate call "
                      "that produced the answer; the prompt, the "
                      "feed, and the decoding are unchanged")
YES_STATUS_NOT_YESNO = ("not a yes/no ask: this door exposes scores, "
                        "but this question form has no yes token "
                        "to score")
YES_STATUS_MISSING = ("yes/no ask, but no score was recorded for "
                      "this row; the words or the error in "
                      "raw_answer are the record")

rows = []
failed = 0

def save_rows():
    out_fields = FIELDS
    for out_path in ("/content/drive/MyDrive/project-k-clips/series2_ola_results.csv",
                     "/content/series2_ola_results.csv"):
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
    want_yes = prompt.rstrip().endswith("Answer yes or no.")
    raw_text = ""
    error = ""
    proof = None
    yp = None
    try:
        raw, proof, yp = ask(job, prompt, max_tokens)
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
    if want_yes:
        if yp is not None:
            ys, yss = f"{yp:.4f}", YES_STATUS_EXPOSED
        else:
            ys, yss = "", YES_STATUS_MISSING
    else:
        ys, yss = "n/a", YES_STATUS_NOT_YESNO
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
        "yes_score": ys,
        "yes_score_status": yss,
        "model": (f"{MODEL_NAME} ({LOAD_MODE}) "
                  "[venv py3.10, torch 2.1.2+cu121]"),
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
## Step 3 · Write the runner, then check: download, ladder, sanity

The model never runs inside this notebook's session. It runs in a
plain script, `/content/ola_series2_run.py`, under the private
environment's Python, so every import it makes resolves inside the
environment Step 1 proved. The first cell below writes that
script. The second runs it in **check mode**.

Check mode does four things, in order:

1. **Proof again.** The versions and files it imported, the
   flash_attn and deepspeed imports the model's code demands, the
   ola package's builder path, and one new line: the yes-score
   fact. The model's own generate method forwards every extra
   keyword to the standard generate, so this door exposes its
   per-step scores on request, and the proof prints that before
   anything loads.
2. **The download.** The model repo `THUdyh/Ola-7b` at the pinned
   revision `d2fcaa533792480ce748321bea5101fed29d857e`, into a
   plain local folder. The runner then rewrites the two speech
   encoder paths in that copy's config to absolute paths, which
   is the model card's own documented step 2: the config ships
   pointing at `./large-v3.pt` and
   `./BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt`, and both
   files are inside the repo, so the rewrite just tells the
   truth about where they are. The authors' code was cloned and
   commit-checked in Step 1
   (`f681820b32db398e62d430c4e5b4f3dd962bbed4`); the runner
   quotes that commit in its load log.
3. **The ladder.** Three rungs, all the same model, loaded
   through the authors' own `load_pretrained_model`, exactly as
   the Series One OLA runner loads it. **Rung 1: full form**,
   16-bit (bfloat16), the authors' default finish. The main
   weights are about 18 GB and the two speech encoders add about
   1.5 GB, so this rung wants about 19 to 20 GB of the L4's 24
   and should load. **Rung 2: the builder's 8-bit squeeze.
   Rung 3: the builder's 4-bit squeeze.** A failed rung prints
   its full traceback, never a cut-short reason, and appends the
   same words to `series2_ola_load_log.txt` in the clips folder,
   so a verbatim record survives the session. If all three fail,
   the runner's closing lines name that file. Whichever rung
   loads is printed and saved with every answer, because a
   squeezed fit is a different run from the full form.
4. **The sanity question.** OLA has no plain text door, so the
   gate rides the full video path on the smallest clip in the
   folder, the beep probe, with one instruction: reply with the
   word ready. The answer prints in full. If it comes back
   empty, with an error, or without the word ready in it, the
   script stops and Step 4 must not be run. This gate exists
   because of the first Qwen 3B run, where every ask died quietly
   and the errors were stored as empty answers.
"""))

cells.append(code(
    "# --- Write the Series Two runner script ----------------------------------------\n"
    "# The runner holds the download, the ladder, the sanity check,\n"
    "# the strict feed proof, and the diagnostic battery. It is\n"
    "# written to a plain file here and only ever run under the\n"
    "# private environment's Python, never imported into this\n"
    "# session.\n"
    "RUNNER_SOURCE = " + repr(RUNNER_SRC) + "\n"
    "with open(\"/content/ola_series2_run.py\", \"w\") as fh:\n"
    "    fh.write(RUNNER_SOURCE)\n"
    "print(\"runner written:\", \"/content/ola_series2_run.py\", "
    "len(RUNNER_SOURCE), \"chars\")\n"))

cells.append(code(r"""
# --- Check mode: proof, download, ladder, sanity ---------------------------------
# Runs the runner under the private environment's Python and streams
# its output here live. A nonzero exit code means the proof, the
# download, the ladder, or the sanity gate failed, and the run stops
# in this cell. The first check run downloads about 21 GB of model
# files; that takes a while and is normal.
import os, subprocess
VENV_PY = "/content/ola-venv/bin/python"
proc = subprocess.run([VENV_PY, "/content/ola_series2_run.py", "check"])
print()
print("check exit code:", proc.returncode)
if proc.returncode != 0:
    print("!" * 72)
    print("STOP. THE CHECK RUN FAILED. The runner's own output above says")
    print("where: a proof line, the download, a ladder rung, or the")
    print("sanity question. Do not run Step 4. Show this output to Muse.")
    print("!" * 72)
    raise SystemExit("STOP: runner check failed")
print("check passed: proof lines right, model loaded, sanity answered.")
print("Go on to Step 4.")
"""))

# ---------------------------------------------------------------- step 4
cells.append(md(r"""
## Step 4 · Run the diagnostic battery

The next cell runs the runner in **battery mode**: proof, download,
and ladder again, sanity again, then the three Series Two tests,
the same 87 asks, files, and question forms as the MiniCPM Series
Two run. Every ask prints its FEED PROOF line before the model's
answer is taken. Watch for it: that line is the whole point of
Series Two.

**How this model is asked.** With-picture asks ride the authors'
video path: moviepy takes the sound out of the video itself,
decord samples 64 frames evenly over the whole clip, the authors'
frame selection check is kept exactly as shipped and the kept
count is printed and saved per ask, and the question rides after
the speech token and the image token in the qwen_1_5 template.
Sound-alone asks (the first arm of Test 2) ride the authors' own
audio+text branch of the same inference script: the soundtrack is
taken out of the same file by the same moviepy step, the question
rides as text after the speech token with no image token in the
prompt, and generate is called with modalities=['text']. The
blank zero image that branch passes is encoded by the vision
tower but, by the authors' own input builder, contributes no
tokens to the sequence when the prompt holds no image token.
Decoding is greedy (temperature 0, sampling off) on both doors;
the authors' demo defaults to 0.2, and the battery's standing
rule for every model is temperature 0, so that one setting
differs from the demo on purpose and is stated here. Token caps
per ask are the MiniCPM Series Two job list's own: 300 for the
open question, 32 for presence, 128 for detail, 16 for yes/no,
32 for the timing asks.

**Test 2, the 2x2 (48 asks).** Sounds in a fixed order: speech
first (the baseline), then true silence (the negative control),
then water, then beeps. Each sound is fed two ways: sound alone,
and the same soundtrack with the swap still picture. Each arm is
asked three ways, two repeats each: the open repaired JSON
question (word for word the Series One question), a direct
presence question made for that sound, and a direct detail
question (for speech: "Write down any words spoken in this
clip."; for the others, a short sound-name question, and for
silence an any-sound control).

**Test 3, yes/no scoring (30 asks).** All seven swap soundtracks,
three versions each: sound on with the still picture, muted with
the still picture, and sound on with a plain gray frame. Each
version is asked about its matching sound ("Is there the sound
of X in this clip? Answer yes or no."), the mixed file about both
of its sounds, and two lures ride along: the speech file is asked
about water, and the water file is asked about speech. The
reading is fixed in advance in the Series Two plan: yes not
moving between sound on and muted means the sound never got in;
yes rising while the free answer still says none means the loss
is at reporting; yes staying high on a muted kitchen picture
means the scene is guessing. Here, unlike MiniCPM, the yes/no
rows also carry a measured score: the probability of a yes token
at the first generated position, read from the same greedy
generate call's own returned scores. The words stay the primary
record; the score is the lean behind the word, and it exists
only because this model's door exposes it.

**Test 4, tone-shift timing (9 asks).** The three tone files
from Step 2. Per file, three separate asks: presence (yes or
no), early-or-late ("Does the tone start in the first half or
the second half of the clip?"), and numeric ("At what second
does the tone start? Answer with a number."). Timing gets scored
later, only on files where presence was detected, so a timing
failure can no longer hide inside a hearing failure.

**Hard stops and warnings.** A feed that fails its proof stops
the whole run on the spot (exit code 7), with the rows completed
so far saved first. An ask that errors or comes back empty is
saved with its full error text in `raw_answer` and a WARNING on
screen, never as silence, and the run ends with the RUN PROBLEM
banner (exit code 6). Results save to `series2_ola_results.csv`
in the Drive clips folder, one row per ask, with the full
feed-proof numbers on every row, in the same schema as MiniCPM's
Series Two results.
"""))

cells.append(code(r"""
# --- Battery mode: the Series Two diagnostic tests --------------------------------
# Same runner, same private Python, streamed live. The runner saves
# the CSV itself, to the Drive clips folder and to /content. A
# nonzero exit code means the pre-flight gates failed up front, a
# feed proof hard-stopped the run, or the run finished with broken
# asks (the banners above say which).
import os, subprocess
VENV_PY = "/content/ola-venv/bin/python"
proc = subprocess.run([VENV_PY, "/content/ola_series2_run.py", "battery"])
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

Read this run against MiniCPM's Series Two run, row for row. The
files were identical and the questions were byte-identical, so a
difference is a model difference or a feed difference, and the
feed differences are written down, not hidden. They are:

- **The doors.** MiniCPM was fed one frame per second with the
  matching second of sound, through its unified chat. OLA is fed
  up to 64 frames per clip (minus whatever the authors' selection
  check dropped, counted per row) and the whole clip's sound
  through two encoders at once, speech mel and raw waveform.
  MiniCPM's sound-alone arm was its chat with no picture; OLA's
  is the authors' audio+text branch, a different documented door
  with no image token in the prompt. An OLA hit on sound is a hit
  with every advantage; an OLA miss on sound cannot be blamed on
  a thin feed.
- **The yes scores.** MiniCPM's chat door exposes no score, so
  its yes/no rows are words only. OLA's door exposes the score,
  so its yes/no rows carry the lean behind the word. Compare
  words to words across the two runs; compare scores only inside
  this one.
- **The unit columns.** In MiniCPM's CSV the unit columns count
  one-second units. In this CSV they count OLA's own 30 second
  chunks. The column names match by design; the notebook and the
  feed text say what the unit is in each run.

Then the tests themselves:

- **Feed proof first.** A row is only as good as its proof line.
  Rows after a hard stop do not exist, on purpose: a stopped run
  is a named failure, not a pile of quiet none answers.
- **Test 2 reads as a square.** Sound alone against sound with a
  picture separates the picture's interference; open against
  direct separates the question's suppression. Speech is the
  baseline and silence is the control: a yes on silence is a
  false alarm, and the rate matters as much as the hits.
  MiniCPM's square said the loss sits in the telling, not the
  hearing. OLA's square either agrees or shows a different
  shape, and the pre-registered prediction hangs on the vacuum
  and sizzle cells of Test 3.
- **Test 3 reads by movement.** Compare each sound across its
  three versions before comparing sounds to each other, and read
  the yes score beside the word: a word that flips while the
  score barely moves, or a score that moves while the word does
  not, is itself a finding about the reporting stage.
- **Test 4 reads in order.** Presence first. Early-or-late and
  numeric starts count only where presence was a hit. MiniCPM
  heard all three tones and placed them +2, -1, +2; three tones
  is a small cell, stated as such, for both models.
- **Scope.** Every claim stays capped at this model, this feed,
  these files, this date. The stages in the Series Two plan stay
  hypotheses until an intervention separates them, and these
  three tests are the interventions. The environment is part of
  the record too: a private virtual environment of its own
  (Python 3.10, torch 2.1.2+cu121, the authors' pinned
  transformers 4.43.4), the authors' code at commit
  f681820b32db398e62d430c4e5b4f3dd962bbed4, weights at hub
  revision d2fcaa533792480ce748321bea5101fed29d857e, and the
  environment string is saved in the `model` column of every
  CSV row.

When you finish, the file `series2_ola_results.csv` in your
Drive is the record. Send it back, and the two Series Two runs
get written up side by side, with the prediction scored as it
was written.
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

out = os.path.join(HERE, "project_k_ola_series2_colab.ipynb")
with open(out, "w") as fh:
    json.dump(nb, fh, indent=1)
print("wrote", out, "cells:", len(cells))
