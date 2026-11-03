#!/usr/bin/env python3
"""Build project_k_qwen7b_equal_colab.ipynb.

Equal-battery notebook for Qwen2.5-Omni-7B: the completed Series One
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
BASE_NB = os.path.join(HERE, "project_k_qwen_omni_7b_colab.ipynb")


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

# The Series One Qwen 7B notebook cells are: 0 title, 1 Step 1
# markdown, 2 Step 1 code. Step 1 is reused because the
# environment is deliberately identical: the same session
# installs, the same pins. Two Series Two additions are made,
# both marked where they land: one paragraph in the markdown
# and a proof block at the end of the code, which prints the
# imported transformers version and file and stops the run if
# the 4.52.4 pin is not met, the same way the MiniCPM and OLA
# Series Two notebooks prove their stacks in Step 1.
step1_md = copy.deepcopy(base_nb["cells"][1])
step1_md["source"] = step1_md["source"] + (
    "\n"
    "**Series Two addition.** At the end, the cell proves the\n"
    "stack it just installed: it prints the imported transformers\n"
    "version and the file it loaded from, and stops the run if\n"
    "the version is not the pinned 4.52.4. The other Series Two\n"
    "notebooks prove their stacks the same way. This matters more\n"
    "here than there, because the Qwen model runs under this\n"
    "session's own Python: there is no private environment, so\n"
    "these installed versions are the run's versions.\n"
).splitlines(True)

step1_code = copy.deepcopy(base_nb["cells"][2])
step1_code["source"] = step1_code["source"] + (
    "\n"
    "# --- Series Two addition: prove the installed stack ----------------------\n"
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
# Project K, equal battery: Qwen2.5-Omni-7B runs the completed Series One battery

**Why this notebook exists.** Kobby's equal-tests ruling (Nov 2,
2026): every model goes through the same Series One battery before
any post speaks across models. This model still owes the blackout, a valid bed, and the muted pairs. This notebook runs the
whole battery for Qwen2.5-Omni-7B, identical to the other three
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
Series Two ones, untouched: the Series One loading ladder under the session Python: full `Qwen/Qwen2.5-Omni-7B` in float16, then the official AWQ build, then the 8-bit bitsandbytes squeeze, the talker turned off on every rung, and the processor from the rung's repo. Step 1 installs the stack into the session with transformers pinned at 4.52.4 and proves the pin. The runner's ask path,
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
`equal_qwen7b_results.csv` in the Drive clips folder.
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

# ------------------------------------------------------------- runner script
# The runner is a plain Python script the notebook writes to
# /content/qwen_7b_equal_run.py and runs under the notebook
# session's own Python. That is the Qwen notebooks' own pattern:
# their stack is installed into the session (Step 1), so the
# session interpreter is the environment, and there is no private
# venv to run under, unlike the MiniCPM and OLA notebooks. The
# runner holds the proof, the Series One loading ladder, the
# sanity gate, the strict feed proof, and the Series Two
# equal battery. The battery section (job list, prompts,
# targets, lures) is copied word for word from the MiniCPM
# Series Two runner.
RUNNER_SRC = r"""
# qwen_7b_equal_run.py - the equal battery runner for
# Qwen2.5-Omni-7B. Written by
# project_k_qwen7b_equal_colab.ipynb. It runs under the
# notebook session's own Python: the Qwen notebooks install
# their stack into the Colab session (Step 1), and there is no
# private environment to run under. Modes: "check" runs the
# proof, the loading ladder, and the sanity question, then
# stops. "battery" does all of that and then runs the Series
# Two diagnostic tests and saves the CSV.
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else "check"
print("runner mode:", MODE)
print("runner python:", sys.version.split()[0], sys.executable)

# --- Proof lines, printed before anything loads --------------------------------
# The versions this script actually imported, from the paths it
# actually loaded them, repeated here so every run carries its
# own evidence. The transformers pin is the Series One Qwen pin;
# anything else stops the run before a model is touched.
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
# models/qwen2_5_omni/modeling_qwen2_5_omni.py):
# Qwen2_5OmniForConditionalGeneration.generate (line 4412) is a
# wrapper. It routes every unprefixed keyword through
# shared_kwargs into the thinker's kwargs (lines 4492 and
# 4496). return_audio defaults to self.has_talker (line 4458),
# and disable_talker(), which every rung of the ladder below
# calls exactly as Series One did, sets has_talker to False
# (lines 4355 to 4360). So generate_audio is False (line 4505)
# and the wrapper returns self.thinker.generate(...) untouched
# (lines 4510 to 4513). The thinker is a standard
# GenerationMixin model, so return_dict_in_generate and
# output_scores asked of that same greedy call come back on its
# result as .sequences and .scores, and the first score row is
# the distribution at the first generated position. The flags
# cannot change the prompt, the feed, or the tokens greedy
# decoding chooses; they only keep numbers the call already
# computes. MiniCPM's chat door, by contrast, returns decoded
# text only, which is why its Series Two rows read "not
# exposed". Here the score is exposed by the door itself, so
# YES_SCORE_EXPOSED is True, and yes/no asks record the
# probability of a yes token at the first generated position
# from that call's first score row.
YES_SCORE_EXPOSED = True
print("yes token score exposed by the generate door:", YES_SCORE_EXPOSED,
      "(the wrapper returns the thinker's own generate result when",
      "the talker is off, and that result carries its scores)")

# The same source holds one more fact this runner respects: the
# thinker's max_new_tokens is pre-seeded from the wrapper's own
# thinker_max_new_tokens parameter (default 1024), and the
# merge only fills keys that are missing, so an unprefixed
# max_new_tokens is silently ignored by this wrapper. The
# Series One notebooks passed the cap unprefixed; their answers
# all ended at the end token long before either cap, so their
# record stands. Here the job list's caps are requirements, so
# every generate call below passes its cap as
# thinker_max_new_tokens, the wrapper's documented knob. The
# prompt, the feed, and the greedy decoding are untouched.

# --- The load log ----------------------------------------------------------------
# A failed rung prints its full traceback (never a cut-short
# reason) and the same words are appended, as events happen, to
# equal_qwen7b_load_log.txt, in the Drive clips folder first
# and /content second, so the file is already on disk even if
# the session dies right after a rung does.
import traceback, datetime

LOAD_LOG_PATHS = ["/content/drive/MyDrive/project-k-clips/equal_qwen7b_load_log.txt",
                  "/content/equal_qwen7b_load_log.txt"]

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
log_load("Qwen2.5-Omni-7B Series Two load log - "
         + datetime.datetime.now().isoformat(timespec="seconds")
         + " - runner mode: " + MODE + "\n")

# --- Load the model: the Series One 7B ladder, unchanged -------------------------
# Three rungs, all of them the 7B, in three different fits,
# tried in the Series One order. A rung that fails prints its
# full traceback and the ladder climbs on. If every rung fails,
# the run stops here: no substitute model is ever loaded
# quietly. On every rung the weights load with
# torch_dtype=float16 and device_map="auto", the talker is
# turned off (written answers only, and it is what makes the
# wrapper return the thinker's result directly), and the
# processor comes from the rung's repo.
from transformers import (Qwen2_5OmniForConditionalGeneration,
                          Qwen2_5OmniProcessor, BitsAndBytesConfig)

def free_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

model = None
processor = None
MODEL_NAME = None   # repo name of the model that loaded
LOAD_MODE = None    # the fit that worked: fp16, AWQ, or 8-bit

RUNGS = []
RUNGS.append(("Qwen/Qwen2.5-Omni-7B", "fp16", {}))
RUNGS.append(("Qwen/Qwen2.5-Omni-7B-AWQ", "AWQ", {}))
RUNGS.append(("Qwen/Qwen2.5-Omni-7B", "8-bit",
              {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}))

rung_errors = []
for repo, mode, extra in RUNGS:
    print("=" * 70)
    print(f"rung: {repo} in {mode} form")
    try:
        m = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            repo, torch_dtype=torch.float16, device_map="auto", **extra)
        m.disable_talker()  # written answers only; saves memory
        p = Qwen2_5OmniProcessor.from_pretrained(repo)
        model, processor = m, p
        MODEL_NAME, LOAD_MODE = repo, mode
        print("loaded:", repo, "in", mode, "form")
        break
    except Exception:
        tb = traceback.format_exc()
        rung_errors.append((repo, mode, tb))
        print("this rung failed. Full traceback:")
        print(tb)
        log_load("-" * 70 + "\n")
        log_load("rung failed: " + repo + " in " + mode + " form\n")
        log_load(tb)
        model = None
        processor = None
        free_memory()

print()
if model is None:
    print("!" * 70)
    print("ALL THREE RUNGS FAILED. The 7B did not load in any fit.")
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

# --- Sanity check: can the model write at all? ---------------------------------
# The Series One gate, in its own form: one plain text question
# through the real processor and generate path (chat template,
# processor, generate, decode only the new tokens), with the
# word ready required in the answer. It proves the model can
# write through this path before any clip runs. It does not
# prove sound arrives; the per-ask FEED PROOF in the battery is
# what proves that. An empty answer, a missing ready, or an
# error all stop the run here, loudly. Both modes run it. This
# gate exists because of the 3B's first Series One run, where
# every ask died on memory and the errors were stored as empty
# answers.
try:
    convo = [{"role": "user",
              "content": [{"type": "text",
                           "text": "Reply with the word ready."}]}]
    stext = processor.apply_chat_template(convo, add_generation_prompt=True,
                                          tokenize=False)
    sinputs = processor(text=stext, return_tensors="pt", padding=True)
    sinputs = sinputs.to(model.device)
    with torch.no_grad():
        sout = model.generate(**sinputs, thinker_max_new_tokens=16,
                              do_sample=False)
    snew = sout[:, sinputs.input_ids.shape[1]:]
    sraw = processor.batch_decode(snew, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)[0]
    print("SANITY RAW OUTPUT:", repr(sraw))
    if not sraw.strip():
        print("WARNING: the sanity check came back EMPTY.")
        print("STOP. Show this output to Muse.")
        sys.exit(5)
    if "ready" not in sraw.lower():
        print("WARNING: the sanity answer does not contain the word ready.")
        print("The write path ran, but the model did not follow the one")
        print("instruction it was given. STOP. Show this output to Muse.")
        sys.exit(5)
    print("sanity check passed: the model can write through the real")
    print("processor and generate path, and the word ready is in its")
    print("answer.")
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
# --- Feed proof machinery ---------------------------------------------------------
# The audit rule: the proof is printed from inside the session,
# per ask, before the answer is taken, and it measures what the
# model's own feed path actually decoded for that ask. The ask
# below first calls process_mm_info on the very conversation it
# is about to feed, exactly as the Series One ask path does, and
# the proof measures audios[0], the decoded waveform array that
# call returns and the processor then receives: duration,
# sample count, RMS in dB, nonzero share, and the same numbers
# per one-second slice of that waveform. One second is the
# proof's measuring grid here (the same grid the MiniCPM Series
# Two CSV uses); the processor takes the waveform whole and
# cuts nothing into seconds, so the padding column records only
# how short the final slice falls of a full second, as
# bookkeeping. The sample rate is 16000 by the loader's
# construction (qwen-omni-utils decodes every soundtrack and
# every audio element at SAMPLE_RATE 16000, mono). The frame
# count comes from the video tensor the same call returns. A
# copy of each decoded track is saved beside the results and
# the track is hashed. The hard-stop thresholds are the MiniCPM
# and OLA Series Two runners': a track that should carry sound
# decoding missing, empty, short by more than 0.5 s against the
# file's own duration, or all zeros stops the run; declared
# silence is exempt by name and verified to be zeros.
import csv, hashlib, json, math, re, subprocess
import numpy as np
import soundfile as sf
from qwen_omni_utils import process_mm_info

CLIPS = "/content/clips"
TRACK_DIRS = ["/content/drive/MyDrive/project-k-clips/equal_qwen7b_extracted_tracks",
              "/content/equal_qwen7b_extracted_tracks"]
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

# The feeds, stated exactly. Both ride the model's own processor
# path; the difference between them is the one the notebook's
# title cell states, and it is written into every row.
FEED_WITH_PICTURE = (
    "the Series One ask path, unchanged: the conversation "
    "carries the video file (fps=1, max_pixels=100352, "
    "min_pixels=3136) and the question; qwen-omni-utils "
    "process_mm_info with use_audio_in_video=True checks the "
    "file carries an audio stream (PyAV) and decodes its "
    "soundtrack with librosa at 16 kHz mono, and samples the "
    "frames in-process (decord, torchvision fallback); the "
    "processor builds the text, Whisper feature extractor "
    "audio features, and video inputs; model.generate is "
    "called with use_audio_in_video=True, greedy "
    "(do_sample=False), the token cap passed as "
    "thinker_max_new_tokens (the wrapper's own knob; a plain "
    "max_new_tokens is ignored by this wrapper), and only the "
    "newly written tokens are decoded; frames fed this ask: "
    "{frames}; at most {tokens} new tokens; the proof above "
    "measures audios[0], the exact decoded waveform array "
    "this same process_mm_info call handed to the processor")
FEED_AUDIO_ALONE = (
    "sound alone through the model's own documented audio "
    "input: the conversation carries an audio element naming "
    "the same file and no video element; qwen-omni-utils "
    "process_audio_info decodes an audio element and a video "
    "soundtrack in the same function with the same librosa "
    "call at 16 kHz mono, so the soundtrack measured above "
    "is the same file's soundtrack through the same decoder; "
    "the processor is called with text and audio only (no "
    "images, no videos, no use_audio_in_video), and "
    "model.generate runs without the in-video flag, greedy "
    "(do_sample=False), the token cap passed as "
    "thinker_max_new_tokens; the difference from the "
    "with-picture arm is exactly this: no picture reaches "
    "the model, and the prompt carries an audio token in "
    "place of video tokens; at most {tokens} new tokens; "
    "the proof above measures audios[0], the exact decoded "
    "waveform array this same process_mm_info call handed "
    "to the processor")

def yes_probability(scores):
    # The probability of a yes token at the first generated
    # position, from the first score row of the same generate
    # call that produced the answer. Softmax over the
    # vocabulary, summed over the yes token ids read from the
    # model's own tokenizer at load time.
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
    # the proof dict on success. The feed text is filled in by
    # ask(), which knows which arm is being ridden.
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

def ask(job, prompt, max_tokens):
    # One question about one file, through the model's own
    # processor path. The with-picture arm is the Series One
    # ask, unchanged. The sound-alone arm rides the documented
    # audio element form. Either way the feed proof prints
    # before the answer is taken, and the generate call is the
    # same greedy call, asked to return its scores so yes/no
    # asks can record the yes probability from its first score
    # row.
    path = os.path.join(CLIPS, job["file"])
    if not os.path.exists(path):
        raise FeedProofError(f"feed file missing: {path}")
    want_yes = prompt.rstrip().endswith("Answer yes or no.")
    if job["arm"] == "audio_alone":
        conversation = [{
            "role": "user",
            "content": [
                {"type": "audio", "audio": path},
                {"type": "text", "text": prompt},
            ],
        }]
        text = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(
            conversation, use_audio_in_video=False)
        proof = prove_feed(path, audios, 0, job["arm"],
                           job["expected_sound"],
                           job["expected_silence"])
        proof["feed"] = FEED_AUDIO_ALONE.format(tokens=max_tokens)
        print("FEED PROOF " + json.dumps(proof, sort_keys=True))
        inputs = processor(text=text, audio=audios,
                           return_tensors="pt", padding=True)
        inputs = inputs.to(model.device).to(model.dtype)
        gen_extra = {}
    else:
        conversation = [{
            "role": "user",
            "content": [
                {"type": "video", "video": path, "fps": 1,
                 "max_pixels": 100352, "min_pixels": 3136},
                {"type": "text", "text": prompt},
            ],
        }]
        text = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False)
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
        proof["feed"] = FEED_WITH_PICTURE.format(
            frames=frames_fed, tokens=max_tokens)
        print("FEED PROOF " + json.dumps(proof, sort_keys=True))
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
    # Decode ONLY the tokens the model just wrote, never the
    # whole sequence, exactly as the Series One ask path does.
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
# --- Equal battery ---------------------------------------------------------
# The repaired JSON question, injected at build time from
# pilot2/run_eggs.py, character for character (444 characters).
PROMPT_JSON = __PROMPT_JSON__
assert len(PROMPT_JSON) == 444, f"prompt drifted: {len(PROMPT_JSON)} chars"

print("FEED SETTINGS, with-picture asks:", FEED_WITH_PICTURE.format(
    frames="N", tokens="M"))
print("FEED SETTINGS, sound-alone asks:", FEED_AUDIO_ALONE.format(
    tokens="M"))
print("MODEL:", MODEL_NAME, "in", LOAD_MODE, "form")

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

rows = []
failed = 0

def save_rows():
    out_fields = FIELDS
    for out_path in ("/content/drive/MyDrive/project-k-clips/equal_qwen7b_results.csv",
                     "/content/equal_qwen7b_results.csv"):
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

RUNNER_SRC = RUNNER_SRC.replace("__PROMPT_JSON__", repr(PROMPT_JSON))
RUNNER_SRC = RUNNER_SRC.replace("__PROMPT_P3__", repr(PROMPT_P3))
assert "__PROMPT_" not in RUNNER_SRC, "prompt placeholders survived"

# ---------------------------------------------------------------- step 3
cells.append(md(r"""
## Step 3 · Write the runner, then check: proof, ladder, sanity

The model never runs inside this notebook's cells directly. It
runs in a plain script, `/content/qwen_7b_equal_run.py`, under this session's own Python (Step 1 installed the stack into the session, so the session interpreter is the environment).
The first cell below writes that script. The second runs it in
**check mode**.

Check mode does three things, in order:

1. **Proof again.** The versions and files the runner imported,
   with the stack pins checked a second time, exactly as the
   Series Two runner proved them.
2. **The loading ladder.** the Series One loading ladder under the session Python: full `Qwen/Qwen2.5-Omni-7B` in float16, then the official AWQ build, then the 8-bit bitsandbytes squeeze, the talker turned off on every rung, and the processor from the rung's repo. Step 1 installs the stack into the session with transformers pinned at 4.52.4 and proves the pin. A failed rung prints its
   full traceback, never a cut-short reason, and appends the same
   words to `equal_qwen7b_load_log.txt` in the clips folder, so a verbatim record
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
    "# The runner holds the proof, the loading ladder, the sanity\n"
    "# gate, the strict feed proof, and the equal battery.\n"
    "# It is written to a plain file here and only ever run under\n"
    "# this session's own Python, the interpreter Step 1\n"
    "# installed the stack into.\n"
    "RUNNER_SOURCE = " + repr(RUNNER_SRC) + "\n"
    "with open(\"/content/qwen_7b_equal_run.py\", \"w\") as fh:\n"
    "    fh.write(RUNNER_SOURCE)\n"
    "print(\"runner written:\", \"/content/qwen_7b_equal_run.py\", "
    "len(RUNNER_SOURCE), \"chars\")\n"))

cells.append(code(r"""
# --- Check mode: proof, ladder, sanity -------------------------------------------
# Runs the runner under this session's own Python and streams
# its output here live. A nonzero exit code means the proof, a
# ladder rung, or the sanity gate failed, and the run stops in
# this cell. The first check run downloads the model files
# once; that takes a while and is normal.
import subprocess, sys
proc = subprocess.run([sys.executable, "/content/qwen_7b_equal_run.py", "check"])
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

**How this model is asked.** the conversation carries the video file at one frame per second with frames capped at 100352 pixels, `process_mm_info` carries the video's own sound track in, and the processor builds the inputs. Token caps travel by the generate wrapper's own thinker setting, as Step 3 of the Series Two notebook explains. Decoding is greedy
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
RUN PROBLEM banner (exit code 6). Results save to `equal_qwen7b_results.csv` in
the Drive clips folder, one row per ask, with the full
feed-proof numbers on every row, in the same 33-column schema
as the Series Two results. The yes-score columns read n/a on
every row: none of these questions is a yes/no ask.
"""))

cells.append(code(r"""
# --- Battery mode: the equal battery tests --------------------------------
# Same runner, same session Python, streamed live. The runner
# saves the CSV itself, to the Drive clips folder and to
# /content. A nonzero exit code means the pre-flight gates
# failed up front, a feed proof hard-stopped the run, or the
# run finished with broken asks (the banners above say which).
import subprocess, sys
proc = subprocess.run([sys.executable, "/content/qwen_7b_equal_run.py", "battery"])
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
the conversation carries the video file at one frame per second with frames capped at 100352 pixels, `process_mm_info` carries the video's own sound track in, and the processor builds the inputs. Token caps travel by the generate wrapper's own thinker setting, as Step 3 of the Series Two notebook explains.

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

When you finish, the file `equal_qwen7b_results.csv` in your Drive is the
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
        "colab": {"gpuType": "T4"},
    },
    "cells": cells,
}

out = os.path.join(HERE, "project_k_qwen7b_equal_colab.ipynb")
with open(out, "w") as fh:
    json.dump(nb, fh, indent=1)
print("wrote", out, "cells:", len(cells))
