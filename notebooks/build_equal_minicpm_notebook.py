#!/usr/bin/env python3
"""Build project_k_minicpm_equal_colab.ipynb.

Equal-battery notebook for MiniCPM-o 2.6: the completed Series One
battery, identical for every open model, per Kobby's
equal-tests ruling of Nov 2, 2026 (see
series-one-2026-11-01/EQUAL-TESTS.md).

It is derived from this model's Series Two notebook builder.
Step 1 (environment and proof) and Step 3 (load path,
precision, sanity gate) are reused unchanged, and the runner's
ask path and feed handling are the Series Two ones. What
changes is Step 2's file list and the battery itself: 41 asks
(Part A the repaired JSON question on the real clips and the
swap set, Part A2 the same question on the real clips muted
and blacked out, Part B the plain sounds question on the
matched bed files, Part B door a transcription question on
the same files, with bed validity judged per prompt).

The questions are not retyped. The repaired JSON question is
pulled from pilot2/run_eggs.py at build time and asserted at
444 characters; the plain sounds question is pulled from
pilot2/swap/run_swap.py and asserted at 143 characters; the
whole job list section is one identical text in all four
equal-battery builders, so the runs ask byte-identical
questions by construction.

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

def extract_p3(path):
    src = open(path).read()
    m = re.search(r"'P3': \((.*?)\)", src, re.S)
    if not m:
        raise SystemExit(f"could not find P3 in {path}")
    ns = {}
    exec("P3 = (" + m.group(1) + ")", ns)
    return ns["P3"]


PROMPT_P3 = extract_p3(os.path.join(PILOT2, "swap", "run_swap.py"))
assert len(PROMPT_P3) == 143, f"P3 drifted: {len(PROMPT_P3)}"



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
# Project K, equal battery: MiniCPM-o 2.6 runs the completed Series One battery

**Why this notebook exists.** Kobby's equal-tests ruling (Nov 2,
2026): every model goes through the same Series One battery before
any post speaks across models. This model still owes the blackout, a valid bed, and the muted pairs. This notebook runs the
whole battery for MiniCPM-o 2.6, identical to the other three
equal-battery notebooks: same files, same questions byte for byte,
same repeats, same 33-column results schema as the Series Two runs.

**The battery (41 asks).**

- **Part A (20 asks).** The repaired JSON question on the three
  real clips (vacuum, jar, eggs) and all seven swap files, two
  repeats each.
- **Part A2 (12 asks).** The same question on the three real
  clips in two built versions: muted (picture, no sound) and
  blackout (sound, no picture), two repeats each.
- **Part B (6 asks).** The plain sounds question on the three
  matched bed files (the sentence alone, the sentence over beeps,
  the sentence over the vacuum bed), two repeats each.
- **Part B door (3 asks).** A transcription question on the same
  three matched files, one repeat each. The bed is judged per
  prompt: a model's bed rows are read only through prompts whose
  baseline (the sentence alone) returned the sentence. The door
  gives every model that baseline check in its strongest form.

**What is reused, unchanged.** Step 1 and Step 3 are this model's
Series Two ones, untouched: the sealed private environment from the Series One notebook: `openbmb/MiniCPM-o-2_6` at its pinned hub revision, loaded on the ladder bf16, then 8-bit, then 4-bit, through the model's own unified chat. The runner's ask path,
feed handling, and feed proof are the Series Two ones. Only Step
2's file list and the battery are new.

**The feed proof rule.** Before every answer is taken, the runner
prints the proof for that exact feed, measured on the model's own
feed path. If a track that should carry sound decodes missing,
empty, short, or all zeros, the run hard-stops with a loud banner
and the rows so far are saved. Declared silence (the swap silence
file and the muted versions) is exempt by name in the job list and
is itself verified to be zeros. Nothing is padded quietly.

**How to run it.** Run the cells from the top, in order. Step 1
builds the environment and proves the stack. Step 2 copies the
thirteen files, builds the six muted and blackout versions,
measures them, and saves the built files into the Drive clips
folder. Step 3 writes the runner and runs its check mode (proof,
loading ladder, sanity). Step 4 runs the battery and saves
`equal_minicpm_results.csv` in the Drive clips folder.
"""))
cells.append(step1_md)
cells.append(step1_code)

# ---------------------------------------------------------------- step 2
cells.append(md(r"""
## Step 2 · Get the clips; build the muted and blackout versions

Everything this run asks about lives in your Google Drive, in the
folder named `project-k-clips`. The next cell links your Drive,
finds that folder, and copies thirteen files to a plain work
folder here:

**The three real clips:**

1. `P02_07_5.mp4`: the vacuum clip.
2. `P32_01_54.mp4`: the jar clip.
3. `own-01.mp4`: your eggs recording.

**The seven swap files** (one still picture, seven soundtracks,
loudness matched at build time): `swap_silence.mp4`,
`swap_speech.mp4`, `swap_beeps.mp4`, `swap_water.mp4`,
`swap_vacuum.mp4`, `swap_sizzle.mp4`, `swap_mixed.mp4`.

**The three matched bed files:**

11. `matched_speech.mp4`: the spoken sentence alone, the baseline.
12. `matched_speech_beeps.mp4`: the sentence over a beep bed.
13. `matched_speech_vacuum.mp4`: the sentence over a vacuum bed.

If any of the thirteen is missing, the cell stops and names it.
Nothing is substituted for a missing file, the matched files
included: a bed test without its exact files is not the bed test.

**Standing rule:** a real clip longer than 20 seconds is cut to
its first 15 seconds (that is the eggs recording), the same
treatment as the earlier notebooks. The swap and matched files
are exact test files and are never cut.

**Then the cell builds six versions** from the real clips as they
stand in the work folder, and measures each one from its decoded
track before printing the measurement:

- **Muted** (`equal_muted_<name>`): the same picture, a silent
  track. The decoded track must measure as silence (RMS at or
  below -90 dB) at the source's duration, or the cell stops.
- **Blackout** (`equal_blackout_<name>`): a plain black picture
  at the clip's own size and length, the original sound
  untouched. The decoded track's duration and RMS must match the
  source clip's (within 0.5 seconds and 1.5 dB), or the cell
  stops.

Built files are saved back into the Drive clips folder, so every
model's run can ride the identical files.

Do not go on until the ready line names all of it: 3 real clips,
7 swap files, 3 matched files, 6 built versions.
"""))

cells.append(code(r"""
# --- Get the clips; build the muted and blackout versions -----------------------
# Copies thirteen files from the Drive clips folder, applies the
# standing cut rule to the real clips, then builds the six muted
# and blackout versions and measures every built file from its
# decoded track. A missing file stops the cell and names the
# file; a built file that decodes wrong stops the cell too.
import os, shutil, subprocess, json, math
import numpy as np

REAL = ["P02_07_5.mp4", "P32_01_54.mp4", "own-01.mp4"]
SWAP = ["swap_silence.mp4", "swap_speech.mp4", "swap_beeps.mp4",
        "swap_water.mp4", "swap_vacuum.mp4", "swap_sizzle.mp4",
        "swap_mixed.mp4"]
MATCHED = ["matched_speech.mp4", "matched_speech_beeps.mp4",
           "matched_speech_vacuum.mp4"]
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

missing = [name for name in REAL + SWAP + MATCHED if not bring(name)]
if missing:
    print("!" * 72)
    print("STOP: these files are missing from the Drive clips folder:")
    for name in missing:
        print("MISSING:", name)
    print("Nothing is substituted for a missing file. Add the exact")
    print("files to the project-k-clips folder, then run this cell again.")
    print("!" * 72)
    raise SystemExit("STOP: files missing in Step 2: "
                     + ", ".join(missing))

def duration_s(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path],
        capture_output=True, text=True)
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
    if os.path.isdir(DRIVE_CLIPS):
        shutil.copy(local_path, os.path.join(
            DRIVE_CLIPS, os.path.basename(local_path)))
        print("saved to Drive clips folder:",
              os.path.basename(local_path))

def video_size(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height",
         "-of", "csv=p=0", path],
        capture_output=True, text=True)
    try:
        w, h = out.stdout.strip().split(",")
        return int(w), int(h)
    except Exception:
        return None, None

# The standing cut rule, real clips only: longer than 20 seconds
# means cut to the first 15. Swap and matched files are exact
# test files and are never cut.
for name in REAL:
    path = os.path.join(WORK, name)
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

build_log = []

# ---- muted versions -------------------------------------------------------------
for name in REAL:
    src = os.path.join(WORK, name)
    out = os.path.join(WORK, "equal_muted_" + name)
    if not os.path.exists(out):
        proc = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", src,
             "-f", "lavfi", "-i",
             "anullsrc=channel_layout=stereo:sample_rate=48000",
             "-map", "0:v:0", "-map", "1:a:0",
             "-c:v", "copy", "-c:a", "aac", "-ar", "48000",
             "-shortest", out])
        if proc.returncode != 0 or not os.path.exists(out):
            raise SystemExit(f"STOP: muted build failed for {name}")
    x, sr = decode_mono(out)
    entry = {"file": os.path.basename(out), "kind": "muted",
             "duration_s": round(len(x) / sr, 3),
             "audio_rms_db_decoded": round(rms_db(x), 2),
             "nonzero_share": round(float(np.count_nonzero(x) / len(x)), 4)}
    src_d = duration_s(src)
    if (src_d is None or abs(entry["duration_s"] - src_d) > 0.5
            or entry["audio_rms_db_decoded"] > -90.0):
        raise SystemExit(f"STOP: muted build for {name} decodes "
                         f"wrong: {entry}")
    build_log.append(entry)
    save_both(out)
    print("muted:", entry)

# ---- blackout versions ------------------------------------------------------------
for name in REAL:
    src = os.path.join(WORK, name)
    out = os.path.join(WORK, "equal_blackout_" + name)
    w, h = video_size(src)
    d = duration_s(src)
    if w is None or d is None:
        raise SystemExit(f"STOP: cannot read size or duration of {name}")
    if not os.path.exists(out):
        proc = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
             "-i", f"color=c=black:s={w}x{h}:r=25:d={d:.3f}",
             "-i", src, "-map", "0:v", "-map", "1:a",
             "-t", f"{d:.3f}", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "48000",
             "-shortest", out])
        if proc.returncode != 0 or not os.path.exists(out):
            raise SystemExit(f"STOP: blackout build failed for {name}")
    xs, srs = decode_mono(src)
    x, sr = decode_mono(out)
    entry = {"file": os.path.basename(out), "kind": "blackout",
             "duration_s": round(len(x) / sr, 3),
             "audio_rms_db_decoded": round(rms_db(x), 2),
             "source_rms_db_decoded": round(rms_db(xs), 2),
             "nonzero_share": round(float(np.count_nonzero(x) / len(x)), 4)}
    if (abs(entry["duration_s"] - d) > 0.5
            or abs(entry["audio_rms_db_decoded"]
                   - entry["source_rms_db_decoded"]) > 1.5):
        raise SystemExit(f"STOP: blackout build for {name} decodes "
                         f"wrong: {entry}")
    build_log.append(entry)
    save_both(out)
    print("blackout:", entry)

if build_log:
    log_path = os.path.join(WORK, "equal_build_log.json")
    with open(log_path, "w") as fh:
        json.dump(build_log, fh, indent=2)
    if os.path.isdir(DRIVE_CLIPS):
        shutil.copy(log_path, os.path.join(DRIVE_CLIPS,
                                           "equal_build_log.json"))
    print("builds logged:", log_path)

print()
ready = True
for name in REAL + SWAP + MATCHED:
    path = os.path.join(WORK, name)
    if os.path.exists(path):
        print("ready:  ", name, f"({os.path.getsize(path) / 1e6:.1f} MB)")
    else:
        ready = False
        print("MISSING:", name)
for name in REAL:
    for kind in ("muted", "blackout"):
        path = os.path.join(WORK, f"equal_{kind}_" + name)
        if os.path.exists(path):
            print("ready:  ", f"equal_{kind}_" + name,
                  f"({os.path.getsize(path) / 1e6:.1f} MB)")
        else:
            ready = False
            print("MISSING:", f"equal_{kind}_" + name)
if not ready:
    print()
    print("!" * 72)
    print("STOP. Some files are missing. Check the Drive folder name is")
    print("exactly project-k-clips, then run this cell again.")
    print("Do not go on to Step 3.")
    print("!" * 72)
    raise SystemExit("STOP: files missing in Step 2")
print()
print("ready: 3 real clips, 7 swap files, 3 matched files, "
      "6 built versions (3 muted, 3 blackout)")
"""))

# ---------------------------------------------------------------- runner
RUNNER_SRC = r"""
# minicpm_equal_run.py - the equal battery runner for MiniCPM-o 2.6.
# Written by project_k_minicpm_equal_colab.ipynb. Run only under the
# private environment's Python (/content/minicpm-venv/bin/python).
# Modes: "check" runs the proof, the loading ladder, and the sanity
# question, then stops. "battery" does all of that and then runs the
# equal battery tests and saves the CSV.
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

LOAD_LOG_PATHS = ["/content/drive/MyDrive/project-k-clips/equal_minicpm_load_log.txt",
                  "/content/equal_minicpm_load_log.txt"]

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
    print("The equal battery was not run. Run Step 4 for the tests.")
    sys.exit(0)
"""

RUNNER_SRC += r"""
# --- Equal battery ---------------------------------------------------------
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
TRACK_DIRS = ["/content/drive/MyDrive/project-k-clips/equal_extracted_tracks",
              "/content/equal_extracted_tracks"]
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

# --- Test definitions: the equal battery (Series One, completed) -----------------
# The plain sounds question, P3, injected at build time from
# pilot2/swap/run_swap.py, character for character (143 characters).
PROMPT_P3 = __PROMPT_P3__
assert len(PROMPT_P3) == 143, f"P3 drifted: {len(PROMPT_P3)} chars"

# The transcription door for the bed test. New for this battery,
# and the same words in all four equal-battery runners: bed
# validity is judged per prompt, and this door gives every model
# the baseline check in its strongest form.
TRANSCRIPT_PROMPT = "Write down any words spoken in this clip. If no words are spoken, say 'no words'."

SWAP_FILES = {
    "silence": "swap_silence.mp4",
    "speech": "swap_speech.mp4",
    "beeps": "swap_beeps.mp4",
    "water": "swap_water.mp4",
    "vacuum": "swap_vacuum.mp4",
    "sizzle": "swap_sizzle.mp4",
    "mixed": "swap_mixed.mp4",
}

REAL_CLIPS = [("vacuum", "P02_07_5.mp4"), ("jar", "P32_01_54.mp4"),
              ("eggs", "own-01.mp4")]

MATCHED_FILES = [("matched_speech", "matched_speech.mp4"),
                 ("matched_speech_beeps", "matched_speech_beeps.mp4"),
                 ("matched_speech_vacuum", "matched_speech_vacuum.mp4")]

def base_job(test, sound, file, arm, version, expected_sound):
    return {"test": test, "sound": sound, "file": file, "arm": arm,
            "version": version,
            "expected_sound": bool(expected_sound),
            "expected_silence": not bool(expected_sound)}

JOBS = []

# Part A: the repaired JSON question on the three real clips and
# all seven swap files, two repeats each (20 asks).
for sound, file in REAL_CLIPS:
    for rep in (1, 2):
        job = base_job("partA_series_one", sound, file,
                       "audio_with_video", "original", True)
        JOBS.append((job, "open", "open_json", sound, PROMPT_JSON,
                     rep, 300))
for sound in ("silence", "speech", "beeps", "water", "vacuum",
              "sizzle", "mixed"):
    for rep in (1, 2):
        job = base_job("partA_series_one", sound, SWAP_FILES[sound],
                       "audio_with_video", "original_swap_still",
                       sound != "silence")
        JOBS.append((job, "open", "open_json", sound, PROMPT_JSON,
                     rep, 300))

# Part A2: the same question on the three real clips muted and
# blacked out, two repeats each (12 asks). The muted job declares
# silence; the blackout job still carries the sound.
for sound, file in REAL_CLIPS:
    for version, expected in (("muted", False), ("blackout", True)):
        for rep in (1, 2):
            job = base_job("partA2_muted_blackout", sound,
                           f"equal_{version}_{file}",
                           "audio_with_video", version, expected)
            JOBS.append((job, "open", "open_json", sound, PROMPT_JSON,
                         rep, 300))

# Part B: the plain sounds question on the three matched bed
# files, two repeats each (6 asks). The first file is the
# baseline, the sentence alone; this prompt's bed rows are read
# only if this prompt's baseline returned the sentence.
for sound, file in MATCHED_FILES:
    for rep in (1, 2):
        job = base_job("partB_matched_bed", sound, file,
                       "audio_with_video", "original", True)
        JOBS.append((job, "direct", "plain_sounds", "speech",
                     PROMPT_P3, rep, 128))

# Part B door: the transcription question on the same three
# matched files, one repeat each (3 asks). Same reading rule:
# the door's bed rows are read only if the door's baseline
# returned the sentence.
for sound, file in MATCHED_FILES:
    job = base_job("partB_door_transcription", sound, file,
                   "audio_with_video", "original", True)
    JOBS.append((job, "direct", "transcription", "speech",
                 TRANSCRIPT_PROMPT, 1, 128))

print(f"equal battery jobs: {len(JOBS)} asks "
      f"(part A: 20, part A2: 12, part B: 6, part B door: 3)")

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
    for out_path in ("/content/drive/MyDrive/project-k-clips/equal_minicpm_results.csv",
                     "/content/equal_minicpm_results.csv"):
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
RUNNER_SRC = RUNNER_SRC.replace("__PROMPT_P3__", repr(PROMPT_P3))
assert "__PROMPT_" not in RUNNER_SRC, "prompt placeholders survived"

# ---------------------------------------------------------------- step 3
cells.append(md(r"""
## Step 3 · Write the runner, then check: proof, ladder, sanity

The model never runs inside this notebook's cells directly. It
runs in a plain script, `/content/minicpm_equal_run.py`, under the private environment Step 1 built, through its own Python (VENV_PY).
The first cell below writes that script. The second runs it in
**check mode**.

Check mode does three things, in order:

1. **Proof again.** The versions and files the runner imported,
   with the stack pins checked a second time, exactly as the
   Series Two runner proved them.
2. **The loading ladder.** the sealed private environment from the Series One notebook: `openbmb/MiniCPM-o-2_6` at its pinned hub revision, loaded on the ladder bf16, then 8-bit, then 4-bit, through the model's own unified chat. A failed rung prints its
   full traceback, never a cut-short reason, and appends the same
   words to `equal_minicpm_load_log.txt` in the clips folder, so a verbatim record
   survives the session. Whichever rung loads is printed and
   saved with every answer.
3. **The sanity question.** The Series Two gate in its own form:
   one plain text question through the real ask path, and the
   word ready is required in the answer. If the answer comes
   back empty, without the word, or with an error, the script
   stops and Step 4 must not be run. Sound arrival is not this
   gate's job; Step 4's per-ask feed proof carries that.

The ladder, the sanity gate, and the ask path are this model's
Series Two ones, unchanged. Only the battery in Step 4 is new.
"""))

cells.append(code(
    "# --- Write the equal battery runner script ----------------------------------------\n"
    "# The runner holds the ladder, the sanity check, the strict feed\n"
    "# proof, and the equal battery. It is written to a plain file\n"
    "# here and only ever run under the private environment's Python.\n"
    "RUNNER_SOURCE = " + repr(RUNNER_SRC) + "\n"
    "with open(\"/content/minicpm_equal_run.py\", \"w\") as fh:\n"
    "    fh.write(RUNNER_SOURCE)\n"
    "print(\"runner written:\", \"/content/minicpm_equal_run.py\", "
    "len(RUNNER_SOURCE), \"chars\")\n"))

cells.append(code(r"""
# --- Check mode: proof, ladder, sanity -------------------------------------------
import os, subprocess
VENV_PY = "/content/minicpm-venv/bin/python"
proc = subprocess.run([VENV_PY, "/content/minicpm_equal_run.py", "check"])
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
## Step 4 · Run the equal battery

The next cell runs the runner in **battery mode**: proof and
ladder again, sanity again, then the 41 asks of the equal
battery, the completed Series One battery, identical in all
four equal-battery notebooks. Every ask prints its FEED PROOF
line before the model's answer is taken. Watch for it: that
line is the discipline that makes the answers evidence.

**How this model is asked.** one frame per second with the matching second of sound as unified chat units, frames downscaled to fit 448x448, at most 15 units, no generated speech. Decoding is greedy
everywhere. Token caps per ask are the job list's own: 300 for
the JSON question, 128 for the plain sounds and transcription
questions.

**Part A (20 asks).** The repaired JSON question, word for word
the Series One question, on the three real clips (vacuum, jar,
eggs) and all seven swap files, two repeats each.

**Part A2 (12 asks).** The same question on the three real
clips in the two versions Step 2 built: muted (the picture with
no sound, so any sound the answer reports is invented or
carried by the picture) and blackout (the sound with a black
picture, so the picture cannot help or interfere), two repeats
each.

**Part B (6 asks).** The plain sounds question (the swap
study's P3: "What sounds can you hear in this clip? Describe
only what is audible, without guessing from the pictures. If
the audio is silent, say 'silent'.") on the three matched bed
files, two repeats each.

**Part B door (3 asks).** "Write down any words spoken in this
clip. If no words are spoken, say 'no words'." on the same
three matched files, one repeat each.

**How the bed is read.** Per prompt, never pooled. Each
prompt's bed rows are read only if that same prompt's baseline
(the sentence alone, `matched_speech.mp4`) returned the
sentence. If the plain sounds baseline fails and the
transcription baseline returns the sentence, the bed is read
through the door, and the reverse likewise. Every row stays in
the CSV either way; the rule is applied at write-up time, not
by deleting rows. This replaces the Series One bed design,
where two of three bed tests came out void for want of a
passing baseline: here every model gets the baseline check in
both forms, including its strongest one.

**Hard stops and warnings.** A feed that fails its proof stops
the whole run on the spot (exit code 7), with the rows
completed so far saved first. An ask that errors or comes back
empty is saved with its full error text in `raw_answer` and a
WARNING on screen, never as silence, and the run ends with the
RUN PROBLEM banner (exit code 6). Results save to `equal_minicpm_results.csv` in
the Drive clips folder, one row per ask, with the full
feed-proof numbers on every row, in the same 33-column schema
as the Series Two results. The yes-score columns read n/a on
every row: none of these questions is a yes/no ask.
"""))

cells.append(code(r"""
# --- Battery mode: the equal battery tests --------------------------------
import os, subprocess
VENV_PY = "/content/minicpm-venv/bin/python"
proc = subprocess.run([VENV_PY, "/content/minicpm_equal_run.py", "battery"])
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

Read this run against this model's own Series One and Series
Two answers, and against the other three models' equal-battery
runs. The battery is identical across the four notebooks: same
files, same questions byte for byte, same repeats. A difference
is a model difference or a feed difference, and the feed
differences are written down, not hidden. This model's feed:
one frame per second with the matching second of sound as unified chat units, frames downscaled to fit 448x448, at most 15 units, no generated speech.

- **Part A against Part A2.** Clean minus muted is what the
  sound adds to the answer. Blackout is what the sound does
  when the picture cannot carry the answer: a model that names
  the sound on the clean clip but not on the blackout was
  reading the picture, not hearing the sound.
- **Part B under the baseline rule.** Read each prompt's bed
  rows only where that prompt's baseline returned the sentence
  (Step 4 states the rule). A bed that bends the transcription
  while the report denies the bed is the bed test's signature
  finding, and it only counts where the baseline passed.
- **The fit.** The `model` column of every row names the rung
  that loaded. A squeezed fit's answers are that fit's answers.
- **Scope.** Every claim stays capped at this model, this
  feed, these files, this date.

When you finish, the file `equal_minicpm_results.csv` in your Drive is the
record. Send it back, and the four equal-battery runs get
written up side by side.
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

out = os.path.join(HERE, "project_k_minicpm_equal_colab.ipynb")
with open(out, "w") as fh:
    json.dump(nb, fh, indent=1)
print("wrote", out, "cells:", len(cells))
