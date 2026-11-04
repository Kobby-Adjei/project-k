#!/usr/bin/env python3
"""Build project_k_qwen_codec_colab.ipynb.

Codec-loss control notebook for Qwen2.5-Omni-3B. It reuses the
Series Two Qwen notebook's plumbing exactly where the two runs
share a path: the same Step 1 (the Series One session installs,
same pins, transformers 4.52.4 proved), the same loading
candidates (the 3B in float16 first, the 7B AWQ build second),
the disable_talker call, the same with-picture ask path
(process_mm_info with use_audio_in_video=True, greedy generate,
the cap passed as thinker_max_new_tokens, only the new tokens
decoded), the same yes-probability extraction from the first
score row of the same generate call, and the same feed proof
measured on audios[0], the exact waveform array the processor
receives, with the same hard stop (exit code 7).

What forks, and only this:

1. The battery. 7 sounds x 2 arms x 2 forms x 2 repeats = 56
   asks. (The codec control plan text says 112 beside these
   same factors; the factors multiply to 56, and this build
   follows the factors.) The sounds are the codec-control
   masters built by
   codec-control/build_battery.py. Arm A is the sound inside
   an MP4 (the swap build recipe), fed through the with-picture
   path above. Arm B is the same sound as an uncompressed WAV,
   fed through the model's documented image + audio element
   form: an image element naming the swap still and an audio
   element naming the WAV file, so the soundtrack that reaches
   the processor is decoded by the same librosa call straight
   from the WAV. No MP4 is opened for Arm B audio.
2. The forms. Open is the swap study's P3 question, pulled from
   pilot2/swap/run_swap.py at build time, never retyped. Yes/no
   is the Series Two template with the codec targets.
3. There is no model-answer sanity gate. A past gate misfired
   on a missing file; here the file and feed assertions are
   the only gates. Before any ask, every battery file is
   decoded through the very path its arm will ride, printed
   (path, SHA-256, duration, per-second RMS), and asserted:
   duration 10.0 +/- 0.5 s, non-silent unless the sound is
   declared silence, and for Arm B the waveform the feed path
   returns must hash equal to a direct decode of the same WAV
   in the same session. Any failure hard-stops with exit 7.
4. The run is resume-safe: asks already present in the output
   CSV are skipped, keyed by (sound, arm, form, target, repeat).

The job list is built here, in the builder, and injected into
the runner as a literal, so the count and the prompt strings
are verified at build time, not trusted.

Plain language throughout, no em dashes.
"""
import copy
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT2 = os.path.join(HERE, "..", "pilot2")
BASE_NB = os.path.join(HERE, "project_k_qwen_omni_colab.ipynb")


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

# Step 1 is the Series One Qwen notebook's own, reused the way
# the Series Two Qwen builder reused it: same session installs,
# same pins, plus the stack proof block the Series Two build
# appended (transformers must be exactly 4.52.4).
step1_md = copy.deepcopy(base_nb["cells"][1])
step1_md["source"] = step1_md["source"] + (
    "\n"
    "**Codec control addition.** At the end, the cell proves the\n"
    "stack it just installed: it prints the imported transformers\n"
    "version and the file it loaded from, and stops the run if\n"
    "the version is not the pinned 4.52.4. The Series Two Qwen\n"
    "notebook proves its stack the same way. This matters here\n"
    "because the Qwen model runs under this session's own\n"
    "Python: there is no private environment, so these installed\n"
    "versions are the run's versions.\n"
).splitlines(True)

step1_code = copy.deepcopy(base_nb["cells"][2])
step1_code["source"] = step1_code["source"] + (
    "\n"
    "# --- Codec control addition: prove the installed stack -------------------\n"
    "# The model runs under this same session Python (the Qwen\n"
    "# notebooks have no private environment), so the versions\n"
    "# installed above are the versions the run uses. Prove them\n"
    "# here, and stop if the transformers pin is not met.\n"
    "import transformers\n"
    "print(\"transformers version:\", transformers.__version__)\n"
    "print(\"transformers file:\", transformers.__file__)\n"
    "if transformers.__version__ != \"4.52.4\":\n"
    "    raise SystemExit(\"STOP: transformers is \"\n"
    "                     + transformers.__version__\n"
    "                     + \", not the pinned 4.52.4. Do not go on.\")\n"
    "print(\"stack proof passed: transformers 4.52.4, the Series One pin.\")\n"
).splitlines(True)

cells = []
cells.append(md(r"""
# Project K, codec-loss control: Qwen2.5-Omni-3B

**Why this notebook exists.** Every Project K run so far fed the
models sound that had passed through a lossy codec: Pilot One at
AAC 96k, Pilot Two and the swap files with the swap build recipe.
A lossy codec throws away quiet detail by design. So a fair
objection stands against the whole record: when a model missed
a quiet sound, maybe the file lost it before the model ever had
the chance. This notebook is the control that answers the
objection. It asks the same model the same questions about the
same sounds, twice over: once with the sound inside an MP4,
decoded the way every earlier run decoded it, and once with the
same sound handed over as an uncompressed WAV waveform that
never passed through the codec.

**What this notebook does.** Seven sounds, each exactly 10
seconds, each in two arms, each asked two ways, two repeats:
56 asks.

- **The sounds.** A 1000 Hz tone at the reference level (burst
  RMS -20 dBFS); the same tone 20 dB quieter; three short sharp
  clicks at the quiet level; the spoken sentence (the swap
  study's own speech file, so this control ties back to it);
  the quiet tone mixed under the matched vacuum bed; a quiet
  two-second noise burst, the sizzle-shaped case; and true
  silence.
- **Arm A, the file way.** The sound inside an MP4 built with
  the swap recipe: the same still picture, the native AAC
  encoder. The recipe sets no bitrate and the encoder's rate
  follows the content; this battery's speech file lands at
  about 38 kbps, matching the shipped swap speech file on
  the same soundtrack. Arm A reproduces the recipe, which is
  what every earlier run actually rode.
- **Arm B, the raw way.** The same sound as a WAV master. The
  conversation carries an image element (the swap still, the
  same picture Arm A carries) and an audio element naming the
  WAV file. The model's own tools decode that element with the
  same call they use for a video soundtrack, but the samples
  come from the WAV itself. No MP4 is opened for the audio.
- **The forms.** The open question is the swap study's P3,
  word for word. The yes/no question is the Series Two form,
  "Is there the sound of X in this clip? Answer yes or no.",
  with X matched to the sound. Yes/no rows also carry the yes
  score, read from the same generate call, the way the Series
  Two Qwen runner reads it.

**The gates are files, not answers.** There is no sanity
question in this notebook. A model-answer gate once misfired
on a missing file and stopped a healthy run, so here the
gates are the file and feed assertions, run before the
battery over every file: each file is decoded through the
very path its arm will ride, and the runner prints its path,
SHA-256, duration, and per-second RMS, then asserts the
duration is 10.0 seconds within half a second, asserts the
track is not silent unless the sound is declared silence
(and that declared silence really is zeros), and for Arm B
asserts the waveform the feed path returns hashes equal to
a direct decode of the same WAV in the same session. Any
failure stops the run with exit code 7 before a single ask.
The same proof then repeats per ask, before each answer, on
the exact waveform array the processor receives.

**Resume.** The battery skips any ask already present in
`codec_qwen_results.csv`, so a stopped session can be
continued by running Step 4 again.

**How to read the result** (fixed in advance, from the codec
control plan):

- Caught raw, missed in AAC: the file did it. That sound's
  earlier misses get re-labeled.
- Missed in both: the model did it; the codec objection dies
  for that sound.
- Caught in both: codec loss does not explain that sound.
- Caught in AAC, missed raw: a build bug. Stop and inspect
  before believing anything.

**Where it runs.** On the free Colab account (kadjei1), under
the routing rule: the 3B is a small model, and this is a
control, not a new battery. **How to run it.** Run the cells
from the top, in order. Step 1 sets up and proves the stack.
Step 2 gets the fifteen battery files. Step 3 writes the
runner and runs its check mode (proof, loading candidates,
file assertions). Step 4 runs the battery and saves
`codec_qwen_results.csv`.

**The one rule.** We claim only what we ran: these sounds,
this model, these two feed paths, this run.
"""))
cells.append(step1_md)
cells.append(step1_code)

# ---------------------------------------------------------------- step 2
cells.append(md(r"""
## Step 2 · Get the codec battery files

Everything this run asks about lives in your Google Drive, in
the folder named `codec-control`. The next cell links your
Drive, finds that folder, and copies fifteen files to a plain
work folder here:

- Seven Arm A files, `codec_<sound>_aac.mp4`: the swap still
  picture carrying each sound, built with the swap recipe.
- Seven Arm B masters, `codec_<sound>_raw.wav`: the same
  sounds as uncompressed WAV, 16-bit PCM at 48000 Hz (the
  speech master is stereo, a byte-for-byte copy of the swap
  study's own speech track; the rest are mono).
- `codec_still.png`: the swap still itself, the picture the
  Arm B asks show the model.

The sounds, in battery order: `tone_ref`, `tone_quiet`,
`clicks_quiet`, `speech`, `tone_under_bed`, `burst_quiet`,
`silence`. Levels and hashes for every file are recorded in
the battery's BATTERY.md; the runner re-measures everything
from inside this session in Step 3 before it believes any of
it.

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

# ------------------------------------------------------------- runner script
# The runner is a plain Python script the notebook writes to
# /content/qwen_codec_run.py and runs under the notebook
# session's own Python, the Qwen notebooks' own pattern.
RUNNER_SRC = r"""
# qwen_codec_run.py - the codec-loss control runner for
# Qwen2.5-Omni-3B. Written by project_k_qwen_codec_colab.ipynb.
# It runs under the notebook session's own Python: the Qwen
# notebooks install their stack into the session (Step 1), and
# there is no private environment to run under. Modes: "check"
# runs the proof, the loading candidates, and the battery file
# assertions, then stops. "battery" does all of that and then
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
import qwen_omni_utils
print("qwen_omni_utils file:", qwen_omni_utils.__file__)
if transformers.__version__ != "4.52.4":
    print("STOP: this script is running on transformers",
          transformers.__version__, "and needs exactly 4.52.4,")
    print("the version the Series One Qwen notebooks pin.")
    sys.exit(3)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available()
      else "none")

# The yes-score question, settled from the model's own code
# before the run (transformers 4.52.4,
# models/qwen2_5_omni/modeling_qwen2_5_omni.py), exactly as the
# Series Two Qwen runner settled it: generate is a wrapper that
# returns the thinker's own result when the talker is off, and
# that result carries its per-step scores when asked. Asking
# the same greedy call for return_dict_in_generate and
# output_scores changes nothing about the prompt, the feed, or
# the tokens chosen; it only keeps numbers the call already
# computes. So yes/no asks record the probability of a yes
# token at the first generated position.
YES_SCORE_EXPOSED = True
print("yes token score exposed by the generate door:", YES_SCORE_EXPOSED,
      "(the wrapper returns the thinker's own generate result when",
      "the talker is off, and that result carries its scores)")

# One more wrapper fact this runner respects, from the same
# source: the thinker's max_new_tokens is pre-seeded from the
# wrapper's own thinker_max_new_tokens parameter, and a plain
# max_new_tokens is silently ignored. Every generate call
# below passes its cap as thinker_max_new_tokens.

# --- The load log ----------------------------------------------------------------
import traceback, datetime

LOAD_LOG_PATHS = ["/content/drive/MyDrive/codec-control/codec_qwen_load_log.txt",
                  "/content/codec_qwen_load_log.txt"]

def log_load(text):
    for p in LOAD_LOG_PATHS:
        try:
            with open(p, "a") as fh:
                fh.write(text)
        except Exception:
            pass   # that home is not mounted; the printed copy stands

def log_paths_saved():
    return [p for p in LOAD_LOG_PATHS if os.path.exists(p)]

log_load("=" * 70 + "\n")
log_load("Qwen2.5-Omni-3B codec control load log - "
         + datetime.datetime.now().isoformat(timespec="seconds")
         + " - runner mode: " + MODE + "\n")

# --- Load the model: the Series One candidates, unchanged ----------------------
from transformers import (Qwen2_5OmniForConditionalGeneration,
                          Qwen2_5OmniProcessor)

def free_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

model = None
processor = None
MODEL_NAME = None   # repo name of the model that loaded
LOAD_MODE = None    # the fit that worked

CANDIDATES = [("Qwen/Qwen2.5-Omni-3B", "fp16"),
              ("Qwen/Qwen2.5-Omni-7B-AWQ", "AWQ weights, fp16 load")]

candidate_errors = []
for repo, mode in CANDIDATES:
    print("=" * 70)
    print(f"candidate: {repo} ({mode})")
    try:
        m = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            repo, torch_dtype=torch.float16, device_map="auto")
        m.disable_talker()  # written answers only; saves memory
        p = Qwen2_5OmniProcessor.from_pretrained(repo)
        model, processor = m, p
        MODEL_NAME, LOAD_MODE = repo, mode
        print("loaded:", repo)
        break
    except Exception:
        tb = traceback.format_exc()
        candidate_errors.append((repo, mode, tb))
        print("this candidate failed. Full traceback:")
        print(tb)
        log_load("-" * 70 + "\n")
        log_load("candidate failed: " + repo + " (" + mode + ")\n")
        log_load(tb)
        model = None
        processor = None
        free_memory()

print()
if model is None:
    print("!" * 70)
    print("BOTH CANDIDATES FAILED. No model loaded.")
    saved = log_paths_saved()
    if saved:
        print("The full traceback of every candidate is saved, word")
        print("for word, in:")
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
print("environment: Colab session python (Step 1 installs),",
      "transformers 4.52.4")
if torch.cuda.is_available():
    used = torch.cuda.memory_allocated() / 1e9
    print(f"chip memory already used by the model: {used:.1f} GB")

# The yes token ids, read from the model's own tokenizer (the
# processor's): the first token of "Yes" and of " Yes". A yes
# score is the summed probability of those ids at the first
# generated position.
YES_TOKEN_IDS = []
for _word in ("Yes", " Yes"):
    _enc = processor.tokenizer.encode(_word, add_special_tokens=False)
    if _enc:
        YES_TOKEN_IDS.append(int(_enc[0]))
YES_TOKEN_IDS = sorted(set(YES_TOKEN_IDS))
print("yes token ids for scoring:", YES_TOKEN_IDS,
      [processor.tokenizer.decode([i]) for i in YES_TOKEN_IDS])
"""

RUNNER_SRC += r"""
# --- Feed proof machinery ---------------------------------------------------------
# The audit rule, as in Series Two: the proof is printed from
# inside the session, per ask, before the answer is taken, and
# it measures the exact decoded waveform array process_mm_info
# returns for that ask and hands to the processor: duration,
# sample count, RMS in dB, nonzero share, the same numbers per
# one-second slice, and a hash. A copy of each decoded track is
# saved beside the results. The hard-stop thresholds are the
# Series Two ones: a track that should carry sound decoding
# missing, empty, short by more than 0.5 s against the file's
# own duration, or all zeros stops the run with exit code 7;
# declared silence is exempt by name and is itself verified
# to be zeros. This runner adds the battery's own assertion:
# every battery file's own duration must be 10.0 s within
# 0.5 s, checked in the assertion pass before the battery and
# again inside prove_feed on every ask.
import csv, hashlib, json, math, re, subprocess
import numpy as np
import librosa
import soundfile as sf
from qwen_omni_utils import process_mm_info

CLIPS = "/content/clips"
STILL = os.path.join(CLIPS, "codec_still.png")
TRACK_DIRS = ["/content/drive/MyDrive/codec-control/codec_qwen_extracted_tracks",
              "/content/codec_qwen_extracted_tracks"]
for d in TRACK_DIRS:
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass

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

def wav_canonical(path):
    # The WAV file's own decoded samples, decoded here directly
    # from the WAV (librosa at 16 kHz mono, the same decode the
    # model's tools use), independent of any conversation. Arm B
    # asserts the waveform the feed path returns hashes equal
    # to this. Also returns the native-rate decode (soundfile,
    # downmixed to mono) so the file proof can tie the 16 kHz
    # waveform back to the samples stored in the file.
    canon, sr = librosa.load(path, sr=16000, mono=True)
    canon = np.asarray(canon, dtype=np.float32)
    native, nsr = sf.read(path, dtype="float32", always_2d=True)
    native_mono = native.mean(axis=1)
    return canon, sr, native_mono, nsr

FEED_AAC = (
    "Arm A, the file way, through the Series One ask path, "
    "unchanged: the conversation carries the video file "
    "(fps=1, max_pixels=100352, min_pixels=3136) and the "
    "question; qwen-omni-utils process_mm_info with "
    "use_audio_in_video=True checks the file carries an audio "
    "stream (PyAV) and decodes its AAC soundtrack with librosa "
    "at 16 kHz mono, and samples the frames in-process; the "
    "processor builds the model's inputs; model.generate is "
    "called with use_audio_in_video=True, greedy "
    "(do_sample=False), the token cap passed as "
    "thinker_max_new_tokens, and only the newly written "
    "tokens are decoded; frames fed this ask: {frames}; at "
    "most {tokens} new tokens; the proof measures audios[0], "
    "the exact decoded waveform array this same "
    "process_mm_info call handed to the processor")
FEED_RAW = (
    "Arm B, the raw way: the conversation carries an image "
    "element naming codec_still.png (the swap still, the same "
    "picture the Arm A files carry) and an audio element "
    "naming the WAV file, and no video element; "
    "qwen-omni-utils process_mm_info decodes the audio element "
    "with the same librosa call at 16 kHz mono it uses for a "
    "video soundtrack, but the samples come from the WAV file "
    "itself: no MP4 is opened for the audio and no AAC decode "
    "stands between the master and the model; the processor "
    "is called with text, audio, and images (no videos, no "
    "use_audio_in_video) and model.generate runs greedy "
    "(do_sample=False), the token cap passed as "
    "thinker_max_new_tokens, only the newly written tokens "
    "decoded; at most {tokens} new tokens; the proof measures "
    "audios[0], the exact decoded waveform array this same "
    "process_mm_info call handed to the processor, and its "
    "hash was asserted equal to a direct decode of the same "
    "WAV before the answer was taken")

def yes_probability(scores):
    if not scores:
        return None
    logits = scores[0][0].float()
    probs = torch.softmax(logits, dim=-1)
    total = 0.0
    for i in YES_TOKEN_IDS:
        total += float(probs[i].item())
    return total

def prove_feed(path, audios, frames_fed, arm, expected_sound,
               expected_silence):
    # Measure the waveform array the feed path decoded for this
    # ask. Raises FeedProofError on any failed check; returns
    # the proof dict on success.
    if audios is None or len(audios) == 0 or audios[0] is None:
        raise FeedProofError(
            f"the feed path returned no audio for {path}")
    audio_np = np.asarray(audios[0], dtype=np.float32)
    if audio_np.ndim > 1:
        audio_np = audio_np[:, 0]
    sr = 16000   # the loader's own rate, by construction
    src_dur = media_duration(path)
    if src_dur is None:
        raise FeedProofError(
            f"could not read media duration for {path}")
    if abs(src_dur - 10.0) > 0.5:
        raise FeedProofError(
            f"battery file duration is not 10.0 s for {path}: "
            f"{src_dur:.3f}s")
    track_dur = len(audio_np) / sr if sr else 0.0
    if len(audio_np) == 0:
        raise FeedProofError(f"decoded track is empty for {path}")
    if track_dur < src_dur - 0.5:
        raise FeedProofError(
            f"decoded track is short for {path}: {track_dur:.3f}s "
            f"of track for a {src_dur:.3f}s file")
    overall_rms = round(rms_db(audio_np), 2)
    nonzero_share = float(np.count_nonzero(audio_np) / len(audio_np))
    unit_count = int(math.ceil(len(audio_np) / sr))
    if unit_count <= 0:
        unit_count = 1
    per_unit_rms = []
    per_unit_padding = []
    zero_chunks = 0
    for i in range(unit_count):
        seg = audio_np[i * sr:(i + 1) * sr]
        per_unit_rms.append(round(rms_db(seg), 2))
        per_unit_padding.append(int(sr - len(seg)) if len(seg) < sr
                                else 0)
        if len(seg) and not np.any(seg):
            zero_chunks += 1
    if expected_sound and (nonzero_share == 0.0 or overall_rms <= -119.5
                           or zero_chunks == unit_count):
        raise FeedProofError(
            f"track that should carry sound decoded silent for "
            f"{path}: rms={overall_rms} dB, "
            f"nonzero share={nonzero_share}")
    if expected_silence and nonzero_share > 0.001:
        raise FeedProofError(
            f"declared silence track is not silent for {path}: "
            f"nonzero share={nonzero_share}")
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
        "frame_ffmpeg_rc": ("n/a (frames are decoded in-process "
                            "by qwen-omni-utils via decord or "
                            "torchvision; this path makes no "
                            "ffmpeg call)"),
        "audio_ffmpeg_rc": ("n/a (the soundtrack is decoded "
                            "in-process by qwen-omni-utils via "
                            "librosa; this path makes no ffmpeg "
                            "call; ffprobe is used only to read "
                            "the file's own duration for the "
                            "short-track check)"),
        "media_duration_s": round(src_dur, 3),
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
        "frames_fed": int(frames_fed),
        "arm": arm,
        "unit_note": ("unit columns count 1 s slices of the "
                      "decoded waveform handed to the "
                      "processor; per-unit padding is the "
                      "final slice's shortfall against a full "
                      "second, proof bookkeeping only"),
    }
    return proof

def conversation_for(job, prompt):
    path = os.path.join(CLIPS, job["file"])
    if job["arm"] == "raw_wav":
        return [{
            "role": "user",
            "content": [
                {"type": "image", "image": STILL},
                {"type": "audio", "audio": path},
                {"type": "text", "text": prompt},
            ],
        }]
    return [{
        "role": "user",
        "content": [
            {"type": "video", "video": path, "fps": 1,
             "max_pixels": 100352, "min_pixels": 3136},
            {"type": "text", "text": prompt},
        ],
    }]

def decode_for(job, prompt):
    # Run the feed path for this job without asking anything:
    # process_mm_info on the very conversation the ask will
    # feed, then the proof. Shared by the assertion pass and
    # by ask(), so what is asserted is what is asked on.
    path = os.path.join(CLIPS, job["file"])
    if not os.path.exists(path):
        raise FeedProofError(f"feed file missing: {path}")
    conversation = conversation_for(job, prompt)
    if job["arm"] == "raw_wav":
        audios, images, videos = process_mm_info(
            conversation, use_audio_in_video=False)
        frames_fed = len(images) if images else 0
    else:
        audios, images, videos = process_mm_info(
            conversation, use_audio_in_video=True)
        frames_fed = 0
        if videos is not None and len(videos):
            try:
                frames_fed = int(videos[0].shape[0])
            except Exception:
                frames_fed = 0
    proof = prove_feed(path, audios, frames_fed, job["arm"],
                       job["expected_sound"],
                       job["expected_silence"])
    if job["arm"] == "raw_wav":
        canon, csr, native_mono, nsr = wav_canonical(path)
        canon_hash = waveform_sha256(canon)
        if canon_hash != proof["track_sha256"]:
            raise FeedProofError(
                f"Arm B waveform hash mismatch for {path}: the "
                f"feed path returned {proof['track_sha256']}, a "
                f"direct decode of the WAV gives {canon_hash}")
        proof["wav_canonical_sha256"] = canon_hash
        proof["wav_native_sample_rate"] = int(nsr)
        proof["wav_native_rms_db"] = round(rms_db(native_mono), 2)
        proof["wav_native_duration_s"] = round(len(native_mono) / nsr, 3)
    return conversation, audios, images, videos, proof

def ask(job, prompt, max_tokens):
    # One question, through the model's own processor path.
    # Arm A rides the Series One with-picture ask unchanged.
    # Arm B rides the image + audio element form. Either way
    # the feed proof prints before the answer is taken, and
    # the generate call is the same greedy call, asked to
    # return its scores so yes/no asks can record the yes
    # probability from its first score row.
    conversation, audios, images, videos, proof = decode_for(
        job, prompt)
    want_yes = prompt.rstrip().endswith("Answer yes or no.")
    if job["arm"] == "raw_wav":
        proof["feed"] = FEED_RAW.format(tokens=max_tokens)
        print("FEED PROOF " + json.dumps(proof, sort_keys=True))
        text = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, audio=audios, images=images,
                           return_tensors="pt", padding=True)
        inputs = inputs.to(model.device).to(model.dtype)
        gen_extra = {}
    else:
        proof["feed"] = FEED_AAC.format(
            frames=proof["frames_fed"], tokens=max_tokens)
        print("FEED PROOF " + json.dumps(proof, sort_keys=True))
        text = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, audio=audios, images=images,
                           videos=videos, return_tensors="pt",
                           padding=True, use_audio_in_video=True)
        inputs = inputs.to(model.device).to(model.dtype)
        gen_extra = {"use_audio_in_video": True}
    with torch.no_grad():
        out = model.generate(**inputs,
                             thinker_max_new_tokens=max_tokens,
                             do_sample=False,
                             return_dict_in_generate=True,
                             output_scores=True, **gen_extra)
    if hasattr(out, "sequences"):
        seqs = out.sequences
        scores = getattr(out, "scores", None)
    elif isinstance(out, tuple):
        seqs = out[0]
        scores = None
    else:
        seqs = out
        scores = None
    new_tokens = seqs[:, inputs.input_ids.shape[1]:]
    decoded = processor.batch_decode(
        new_tokens, skip_special_tokens=True,
        clean_up_tokenization_spaces=False)[0].strip()
    yp = yes_probability(scores) if want_yes else None
    del audios, images, videos, inputs, out, seqs, new_tokens
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return decoded, proof, yp
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

YES_STATUS_EXPOSED = ("exposed: the probability of a yes token at the "
                      "first generated position, read from the "
                      "output_scores of the same greedy generate call "
                      "that produced the answer (the model's generate "
                      "wrapper returns the thinker's own result when "
                      "the talker is off, as in this run); the prompt, "
                      "the feed, and the decoding are unchanged")
YES_STATUS_NOT_YESNO = ("not a yes/no ask: this door exposes scores, "
                        "but this question form has no yes token "
                        "to score")
YES_STATUS_MISSING = ("yes/no ask, but no score was recorded for "
                      "this row; the words or the error in "
                      "raw_answer are the record")

OUT_PATHS = ["/content/drive/MyDrive/codec-control/codec_qwen_results.csv",
             "/content/codec_qwen_results.csv"]

def ask_id_of(job, form, target, rep):
    return (f"{job['sound']}|{job['arm']}|{form}|{target}|{rep}")

# --- Battery file assertions: the only gates ---------------------------------------
# Before any ask, every battery file is decoded through the
# very path its arm will ride and asserted. A failure here
# stops the run with exit code 7 before a single question is
# asked. There is no model-answer sanity gate in this runner:
# a past gate misfired on a missing file, so the gates are
# the files themselves.
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
            conversation, audios, images, videos, proof = decode_for(
                job, prompt)
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
        del audios, images, videos
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print()
    print(f"file assertions passed for {len(seen)} files "
          f"(7 sounds x 2 arms).")

assert_battery_files()

if MODE == "check":
    print()
    print("check mode done: proof, candidates, and file assertions")
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
    want_yes = prompt.rstrip().endswith("Answer yes or no.")
    raw_text = ""
    error = ""
    proof = None
    yp = None
    try:
        raw, proof, yp = ask(job, prompt, max_tokens)
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
    if want_yes:
        if yp is not None:
            ys, yss = f"{yp:.4f}", YES_STATUS_EXPOSED
        else:
            ys, yss = "", YES_STATUS_MISSING
    else:
        ys, yss = "n/a", YES_STATUS_NOT_YESNO
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
        "yes_score": ys,
        "yes_score_status": yss,
        "model": (f"{MODEL_NAME} ({LOAD_MODE}) "
                  "[session python, transformers 4.52.4]"),
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
## Step 3 · Write the runner, then check: proof, candidates, file assertions

The model never runs inside this notebook's cells directly. It
runs in a plain script, `/content/qwen_codec_run.py`, under
this session's own Python, the Qwen notebooks' pattern. The
first cell below writes that script. The second runs it in
**check mode**.

Check mode does three things, in order:

1. **Proof again.** The versions and files the runner
   imported, with the transformers pin checked a second time:
   exactly 4.52.4, or the run stops. A line also prints the
   yes-score fact: this model's generate wrapper returns the
   thinker's own result when the talker is off, so the door
   exposes its per-step scores on request.
2. **The loading candidates.** The Series One list, unchanged:
   the 3B in 16-bit form (float16) first, the 7B AWQ build
   second, only if the small one fails. The talker is turned
   off on whichever loads. A failed candidate prints its full
   traceback and appends the same words to
   `codec_qwen_load_log.txt` in the clips folder.
3. **The battery file assertions.** Every one of the fourteen
   battery files is decoded through the very path its arm
   will ride, printed with its SHA-256, duration, and
   per-second RMS, and asserted: 10.0 seconds within half a
   second; sound where sound is declared, true zeros where
   silence is declared; and for the seven WAVs, the waveform
   the feed path returns must hash equal to a direct decode
   of the same WAV. Any failure stops the run here, exit
   code 7, before a single ask.

There is deliberately no sanity question. The gates are the
files, and they are checked on the model's real feed paths,
not on a text-only warm-up.
"""))

cells.append(code(
    "# --- Write the codec control runner script --------------------------------------\n"
    "# The runner holds the proof, the loading candidates, the\n"
    "# battery file assertions, the strict per-ask feed proof,\n"
    "# and the 56 codec asks. It is written to a plain file\n"
    "# here and only ever run under this session's own Python,\n"
    "# the interpreter Step 1 installed the stack into.\n"
    "RUNNER_SOURCE = " + repr(RUNNER_SRC) + "\n"
    "with open(\"/content/qwen_codec_run.py\", \"w\") as fh:\n"
    "    fh.write(RUNNER_SOURCE)\n"
    "print(\"runner written:\", \"/content/qwen_codec_run.py\", "
    "len(RUNNER_SOURCE), \"chars\")\n"))

cells.append(code(r"""
# --- Check mode: proof, candidates, file assertions --------------------------------
# Runs the runner under this session's own Python and streams
# its output here live. A nonzero exit code means the proof, a
# loading candidate, or a battery file assertion failed, and
# the run stops in this cell. The first check run downloads
# the model files once; that takes a while and is normal.
import subprocess, sys
proc = subprocess.run([sys.executable, "/content/qwen_codec_run.py", "check"])
print()
print("check exit code:", proc.returncode)
if proc.returncode != 0:
    print("!" * 72)
    print("STOP. THE CHECK RUN FAILED. The runner's own output above says")
    print("where: a proof line, a loading candidate, or a battery file")
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
candidates, and the file assertions again, then the 56 asks.
Every ask prints its FEED PROOF line before the model's
answer is taken, and an ASK line naming the ask, its arm,
its file, and the RMS of the waveform actually passed.

**How the arms differ, stated plainly.** Arm A rides the
Series One ask path: the conversation carries the MP4, the
tools decode its AAC soundtrack, the processor takes text,
audio, images, and video. Arm B carries an image element
(the swap still) and an audio element (the WAV file); the
tools decode the WAV with the same call, and the processor
takes text, audio, and images only. The picture content is
the same still in both arms. The soundtrack content is the
same sound in both arms. The only difference is what the
sound passed through on its way in: an AAC encode and
decode, or nothing.

**Token caps.** 300 for the open question, 16 for yes/no,
passed through the wrapper's thinker setting, as in Series
Two. Decoding is greedy everywhere.

**Hard stops and warnings.** A feed that fails its proof
stops the whole run on the spot (exit code 7), with the rows
completed so far saved first. An ask that errors or comes
back empty is saved with its full error text in `raw_answer`
and a WARNING on screen, never as silence, and the run ends
with the RUN PROBLEM banner (exit code 6). If the session
dies, run Step 4 again: asks already in
`codec_qwen_results.csv` are skipped.

Results save to `codec_qwen_results.csv` in the Drive clips
folder, one row per ask, with the full feed-proof numbers on
every row, in the Series Two schema plus an ask_id column.
"""))

cells.append(code(r"""
# --- Battery mode: the codec control asks ------------------------------------------
# Same runner, same session Python, streamed live. The runner
# saves the CSV itself, to the Drive clips folder and to
# /content. A nonzero exit code means the pre-flight gates
# failed up front, a feed proof hard-stopped the run, or the
# run finished with broken asks (the banners above say which).
import subprocess, sys
proc = subprocess.run([sys.executable, "/content/qwen_codec_run.py", "battery"])
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

Read the two arms against each other, sound by sound, before
reading sounds against each other. The reading was fixed in
advance:

- **Caught raw, missed in AAC:** the file did it. The codec,
  not the model, lost that sound, and any earlier Project K
  miss on a sound like it has to be re-labeled.
- **Missed in both:** the model did it. The sound arrived
  intact (the Arm B proof says so, hash and all) and was
  still not reported. The codec objection dies for that
  sound.
- **Caught in both:** codec loss does not explain that sound.
- **Caught in AAC, missed raw:** a build bug, by definition
  of the design. Stop and inspect the feed paths before
  believing any row.

Then compare with the Series Two record. The speech arm of
this battery is the swap study's own speech file, so its two
arms here speak directly to the swap results: if speech is
caught raw and missed in AAC, part of the famous speech
denial was the file. If it is missed in both, the denial is
the model's, now on a feed that provably lost nothing.

- **The yes scores.** Words first, scores beside them, as in
  Series Two: a word that flips between arms while the score
  barely moves, or a score that moves while the word does
  not, is itself a finding about where the loss sits.
- **Scope.** Every claim stays capped at this model, these
  sounds, these two feed paths, this date. Arm B proves what
  arrives when nothing is compressed away; it does not prove
  what any deployed system hears, because deployed systems
  ride files, which is Arm A's world, and that is exactly
  why both arms are in the record.

When you finish, the file `codec_qwen_results.csv` in your
Drive is the record. Send it back, and the codec control gets
written up next to the Series Two runs.
"""))

nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "cells": cells,
}

out = os.path.join(HERE, "project_k_qwen_codec_colab.ipynb")
with open(out, "w") as fh:
    json.dump(nb, fh, indent=1)
print("wrote", out, "cells:", len(cells))
