#!/usr/bin/env python3
"""Build project_k_ola_equal_colab.ipynb.

Equal-battery notebook for OLA (7B): the completed Series One
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

# The Series One OLA notebook cells are: 0 title, 1 Step 1 markdown,
# 2 Step 1 code. Step 1 is reused byte for byte because the
# environment, the authors' stack, the build-constraints fix, and
# the proof are deliberately identical.
step1_md = copy.deepcopy(base_nb["cells"][1])
step1_code = copy.deepcopy(base_nb["cells"][2])

cells = []
cells.append(md(r"""
# Project K, equal battery: OLA (7B) runs the completed Series One battery

**Why this notebook exists.** Kobby's equal-tests ruling (Nov 2,
2026): every model goes through the same Series One battery before
any post speaks across models. This model owes the whole Series One battery; the Series Two cells were its only run before this. This notebook runs the
whole battery for OLA (7B), identical to the other three
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
Series Two ones, untouched: the sealed private environment from the Series One notebook: `THUdyh/Ola-7b` at its pinned head revision, loaded by the authors' own `load_pretrained_model` on the ladder bf16, then 8-bit, then 4-bit. The runner's ask path,
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
thirteen battery files plus the sanity probe clip, builds the six muted and blackout versions,
measures them, and saves the built files into the Drive clips
folder. Step 3 writes the runner and runs its check mode (proof,
loading ladder, sanity). Step 4 runs the battery and saves
`equal_ola_results.csv` in the Drive clips folder.
"""))
cells.append(step1_md)
cells.append(step1_code)

# ---------------------------------------------------------------- step 2
cells.append(md(r"""
## Step 2 · Get the clips; build the muted and blackout versions

Everything this run asks about lives in your Google Drive, in the
folder named `project-k-clips`. The next cell links your Drive,
finds that folder, and copies thirteen battery files plus the
sanity probe clip (`probe_tone.mp4`, the beep clip the Step 3
sanity gate asks about) to a plain work folder here:

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

If any of the fourteen is missing, the cell stops and names it.
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
# Copies thirteen battery files plus the sanity probe clip
# (probe_tone.mp4, which the runner's Step 3 sanity gate asks
# about; the Nov 2, 2026 equal run stopped at that gate because
# this list had dropped it) from the Drive clips folder, applies
# the standing cut rule to the real clips, then builds the six muted
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
PROBE = ["probe_tone.mp4"]
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

missing = [name for name in REAL + SWAP + MATCHED + PROBE
           if not bring(name)]
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
# /content/ola_equal_run.py and runs under the private
# environment's Python. It holds the model download, the loading
# ladder, the sanity check, the strict feed proof, and the Series
# Two equal battery. The with-picture ask path is the authors'
# video route from inference/infer.py, lifted into a function as
# in the Series One OLA runner. The sound-alone ask path is the
# same script's audio+text branch, for the reasons and code
# evidence set out in this builder's docstring. The battery section
# (job list, prompts, targets, lures) is copied word for word from
# the MiniCPM Series Two runner.
RUNNER_SRC = r"""
# ola_equal_run.py - the equal battery runner for OLA (7B).
# Written by project_k_ola_equal_colab.ipynb. Run only under the
# private environment's Python (/content/ola-venv/bin/python).
# Modes: "check" runs the proof, the model download, the loading
# ladder, and the sanity question, then stops. "battery" does all
# of that and then runs the equal battery tests and saves
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
# equal_ola_load_log.txt, in the Drive clips folder first and
# /content second, so the file is already on disk even if the
# session dies right after a rung does.
import traceback, datetime, gc

LOAD_LOG_PATHS = ["/content/drive/MyDrive/project-k-clips/equal_ola_load_log.txt",
                  "/content/equal_ola_load_log.txt"]

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
TRACK_DIRS = ["/content/drive/MyDrive/project-k-clips/equal_ola_extracted_tracks",
              "/content/equal_ola_extracted_tracks"]
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
    print("The equal battery was not run. Run Step 4 for the tests.")
    sys.exit(0)
"""

RUNNER_SRC += r"""
# --- Equal battery ---------------------------------------------------------
# The repaired JSON question, injected at build time from
# pilot2/run_eggs.py, character for character (444 characters).
PROMPT_JSON = __PROMPT_JSON__
assert len(PROMPT_JSON) == 444, f"prompt drifted: {len(PROMPT_JSON)} chars"

print("FEED SETTINGS, video asks:", FEED_VIDEO_TEMPLATE.format(
    kept="N", sampled=64, total="T", tokens="M"))
print("FEED SETTINGS, sound-alone asks:", FEED_AUDIO_TEMPLATE.format(
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
    for out_path in ("/content/drive/MyDrive/project-k-clips/equal_ola_results.csv",
                     "/content/equal_ola_results.csv"):
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
RUNNER_SRC = RUNNER_SRC.replace("__PROMPT_P3__", repr(PROMPT_P3))
assert "__PROMPT_" not in RUNNER_SRC, "prompt placeholders survived"

# ---------------------------------------------------------------- step 3
cells.append(md(r"""
## Step 3 · Write the runner, then check: proof, ladder, sanity

The model never runs inside this notebook's cells directly. It
runs in a plain script, `/content/ola_equal_run.py`, under the private environment Step 1 built, through its own Python (VENV_PY).
The first cell below writes that script. The second runs it in
**check mode**.

Check mode does three things, in order:

1. **Proof again.** The versions and files the runner imported,
   with the stack pins checked a second time, exactly as the
   Series Two runner proved them.
2. **The loading ladder.** the sealed private environment from the Series One notebook: `THUdyh/Ola-7b` at its pinned head revision, loaded by the authors' own `load_pretrained_model` on the ladder bf16, then 8-bit, then 4-bit. A failed rung prints its
   full traceback, never a cut-short reason, and appends the same
   words to `equal_ola_load_log.txt` in the clips folder, so a verbatim record
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
    "# The runner holds the download, the ladder, the sanity check,\n"
    "# the strict feed proof, and the equal battery. It is\n"
    "# written to a plain file here and only ever run under the\n"
    "# private environment's Python, never imported into this\n"
    "# session.\n"
    "RUNNER_SOURCE = " + repr(RUNNER_SRC) + "\n"
    "with open(\"/content/ola_equal_run.py\", \"w\") as fh:\n"
    "    fh.write(RUNNER_SOURCE)\n"
    "print(\"runner written:\", \"/content/ola_equal_run.py\", "
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
proc = subprocess.run([VENV_PY, "/content/ola_equal_run.py", "check"])
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
## Step 4 · Run the equal battery

The next cell runs the runner in **battery mode**: proof and
ladder again, sanity again, then the 41 asks of the equal
battery, the completed Series One battery, identical in all
four equal-battery notebooks. Every ask prints its FEED PROOF
line before the model's answer is taken. Watch for it: that
line is the discipline that makes the answers evidence.

**How this model is asked.** the authors' own inference path: frames sampled across the whole clip (their selection check keeps a varying count, recorded per ask) and the whole soundtrack through the model's two audio encoders at once, Whisper large-v3 for speech and BEATs for sound events. Greedy decoding is the one deliberate change from the authors' demo. Decoding is greedy
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
RUN PROBLEM banner (exit code 6). Results save to `equal_ola_results.csv` in
the Drive clips folder, one row per ask, with the full
feed-proof numbers on every row, in the same 33-column schema
as the Series Two results. The yes-score columns read n/a on
every row: none of these questions is a yes/no ask.
"""))

cells.append(code(r"""
# --- Battery mode: the equal battery tests --------------------------------
# Same runner, same private Python, streamed live. The runner saves
# the CSV itself, to the Drive clips folder and to /content. A
# nonzero exit code means the pre-flight gates failed up front, a
# feed proof hard-stopped the run, or the run finished with broken
# asks (the banners above say which).
import os, subprocess
VENV_PY = "/content/ola-venv/bin/python"
proc = subprocess.run([VENV_PY, "/content/ola_equal_run.py", "battery"])
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
the authors' own inference path: frames sampled across the whole clip (their selection check keeps a varying count, recorded per ask) and the whole soundtrack through the model's two audio encoders at once, Whisper large-v3 for speech and BEATs for sound events. Greedy decoding is the one deliberate change from the authors' demo.

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

When you finish, the file `equal_ola_results.csv` in your Drive is the
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

out = os.path.join(HERE, "project_k_ola_equal_colab.ipynb")
with open(out, "w") as fh:
    json.dump(nb, fh, indent=1)
print("wrote", out, "cells:", len(cells))
