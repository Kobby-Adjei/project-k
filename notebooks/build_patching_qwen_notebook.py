#!/usr/bin/env python3
"""Build project_k_qwen_patching_colab.ipynb.

Series Three, test 2, the surgery: activation patching on
Qwen2.5-Omni-3B. Tracing rounds 1 and 2 walked the sound
through the model's layers with helpers (tiny classifiers)
and an attention watch; both are readers only. The surgery
is the third instrument from the tracing plan
(series-two-2026-11-02/tracing-plan.md) and the causal
check on what the readers found: take the hidden states a
donor clip produced at one thinker decoder layer, at the
audio-token positions, and paste them into a recipient
clip's run at the same layer and positions. If the
recipient's answer changes, those numbers carried the
sound. The pre-registered predictions are quoted in the
closing prints, with no verdict printed by the notebook;
the read happens outside.

Design, as specified:

* Same house pattern as the tracing round 2 build: the
  model runs in a plain script under the session's own
  Python (transformers pinned at exactly 4.52.4), check
  mode first with hard stops (stack proof exit 3, load
  exit 4, token check exit 11, patching mechanics exit
  12), feed proof with SHA-256 prefixes against the clips'
  manifest (exit 7), outputs saved to the Drive folder
  tracing-qwen3b and to /content, full tracebacks to a
  readable load log, fixed seed, plain words, no em
  dashes.
* Clips: 20 WAVs in tracing-qwen3b/patching_clips (tone,
  silence, speech; all exactly 4.000 s, 16 kHz mono),
  resolved from Drive in Step 2. Sound alone, no picture.
* The presence questions and the yes scorer are the
  Series Two Qwen runner's own, copied from
  build_series2_qwen_notebook.py (see the citations in
  the spec block below), and the open question is the
  swap study's P3, pulled from pilot2/swap/run_swap.py
  at build time exactly as the codec builder pulls it.
* Baselines for all 20 clips, then the patching sweep:
  tone into silence (8 pairs, layers 0 to 35), speech
  into silence (4 pairs, all layers), and three controls
  (silence into tone, tone into tone as a sham, and tone
  into silence patched at the non-audio positions only).
  Every patched run appends its row to
  patching_results.csv at once, so an interruption loses
  nothing. At the end, the layer with the largest mean
  tone lift gets an open-form check on the 8 patched
  silence runs.
* Attention: this build reads no attention weights, so
  the model loads with the library default attention
  (sdpa), the door the Series Two asks ran through.
  Tracing round 2 loaded eager only because it had to
  return attention weights; nothing here does.
"""
import copy
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT2 = os.path.join(HERE, "..", "pilot2")
BASE_NB = os.path.join(HERE, "project_k_qwen_omni_colab.ipynb")

# ------------------------------------------------------------ the questions
# The presence questions are built here, in the builder,
# from the Series Two Qwen runner's own constructor and
# target table, so the exact strings are verified at build
# time, not trusted:
#   constructor: build_series2_qwen_notebook.py lines
#     1304-1305, prompt = (f"Is there the sound of {target}
#     in this clip? " "Answer yes or no.")
#   targets: the same file's TARGETS table (line 1277),
#     "speech": ["a person speaking"] (line 1279),
#     "beeps": ["beeping"] (line 1280).
# The tone clips are asked with the beeps target and the
# speech clips with the speech target, per the brief; the
# silence clips are asked with the beeps target (they are
# the tone sweep's recipients and the tone question's
# negative control).
TARGET_BEEPS = "beeping"
TARGET_SPEECH = "a person speaking"


def presence_prompt(target):
    return (f"Is there the sound of {target} in this clip? "
            "Answer yes or no.")


PROMPT_BEEPS = presence_prompt(TARGET_BEEPS)
PROMPT_SPEECH = presence_prompt(TARGET_SPEECH)
assert PROMPT_BEEPS == ("Is there the sound of beeping in this "
                        "clip? Answer yes or no."), PROMPT_BEEPS
assert PROMPT_SPEECH == ("Is there the sound of a person speaking "
                         "in this clip? Answer yes or no."), \
    PROMPT_SPEECH

# The open question: the swap study's P3, extracted from
# pilot2/swap/run_swap.py (the 'P3' entry of its PROMPTS
# dict, lines 51-54) at build time, the codec builder's
# own method (build_codec_qwen_notebook.py lines 63-77),
# and asserted against the expected literal.


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

# ------------------------------------------------------------ battery spec
# The counts and the CSV schemas live here as literals so
# the build can be verified without running anything.
SEED = 20261105
EXPECTED_AUDIO_POSITIONS = 100
N_LAYERS_EXPECTED = 36
WRONG_POSITION_LAYERS = [0, 7, 14, 21, 28, 35]
PRESENCE_MAX_TOKENS = 16   # Series Two Test 3 yes/no cap:
                           # build_series2_qwen_notebook.py
                           # line 1306 (the JOBS append for
                           # test3_yes_no passes 16)
OPEN_MAX_TOKENS = 300      # the codec Qwen runner's open P3
                           # cap: build_codec_qwen_notebook.py
                           # line 105, ("open", "open_p3", 300)
TONE_CLIPS = [f"tone_{i:02d}" for i in range(8)]
SILENCE_CLIPS = [f"silence_{i:02d}" for i in range(8)]
SPEECH_CLIPS = [f"speech_{i:02d}" for i in range(4)]
ALL_CLIPS = TONE_CLIPS + SILENCE_CLIPS + SPEECH_CLIPS
assert len(ALL_CLIPS) == 20
RESULTS_FIELDS = ["pair", "donor", "recipient", "direction", "layer",
                  "yes_score", "answer_word"]
BASELINE_FIELDS = ["clip", "question_target", "prompt", "form",
                   "yes_score", "answer"]
OPENFORM_FIELDS = ["pair", "layer", "answer"]
# The sweep's row math, worked out here so the closing
# prints and this build agree:
#   tone_to_silence   8 pairs x 36 layers = 288
#   speech_to_silence 4 pairs x 36 layers = 144
#   silence_to_tone   4 pairs x 36 layers = 144
#   tone_to_tone      4 pairs x 36 layers = 144
#   wrong_positions   4 pairs x  6 layers =  24
#   total                                   744
EXPECTED_RESULTS_ROWS = (8 * 36 + 4 * 36 + 4 * 36 + 4 * 36
                         + 4 * len(WRONG_POSITION_LAYERS))
assert EXPECTED_RESULTS_ROWS == 744


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip().splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip().splitlines(True)}


with open(BASE_NB) as fh:
    base_nb = json.load(fh)

# Step 1 is the Series One Qwen notebook's own, reused the
# way the tracing builders reused it: same session installs,
# same pins, plus the stack proof block (transformers
# exactly 4.52.4). This build trains nothing, so there is no
# scikit-learn proof; the tracing addition's other words are
# kept in spirit and adjusted to say so.
step1_md = copy.deepcopy(base_nb["cells"][1])
step1_md["source"] = step1_md["source"] + (
    "\n"
    "**Patching addition.** At the end, the cell proves the\n"
    "stack it just installed: it prints the imported\n"
    "transformers version and the file it loaded from, and\n"
    "stops the run if the version is not the pinned 4.52.4.\n"
    "This build trains nothing and reads no attention\n"
    "weights, so no further library proof is needed. This\n"
    "notebook's model runs under the session's own Python,\n"
    "so the versions printed here are the run's versions.\n"
).splitlines(True)

step1_code = copy.deepcopy(base_nb["cells"][2])
step1_code["source"] = step1_code["source"] + (
    "\n"
    "# --- Patching addition: prove the installed stack -------------------------\n"
    "# The model runs under this same session Python, so the\n"
    "# versions installed above are the versions the run uses.\n"
    "# Prove them here, and stop if the transformers pin is\n"
    "# not met.\n"
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
# Project K, Series Three: the surgery (activation patching), Qwen2.5-Omni-3B

**Why this notebook exists.** The tracing asked where a
sound gets lost inside the model. The helpers (tiny
classifiers trained on one layer's numbers at a time) only
read; the attention watch only reads. Round 2's readout,
with its controls at chance, said the sound's presence
stays readable at every layer and the answer barely looks
at the sound's positions. Reading cannot prove cause. The
surgery can, and it is the third instrument in the tracing
plan and the causal check on the readers.

**The surgery, in plain words.** Run a clip that carries
the sound (the donor) and save the numbers it produces at
one layer, at the sound's own positions in the row. Then
run a second clip (the recipient) with a hook set at that
same layer: the moment the layer finishes, the hook
overwrites the recipient's numbers at the sound's
positions with the donor's. Everything after that layer
runs on the patched numbers. If a silence run, patched
with a tone run's numbers, now answers yes to the beeping
question, then those numbers were carrying the tone, at
that layer, causally. Sweeping the layer from early to
late brackets where the sound stops being able to change
the answer.

**The design, fixed in advance.**

- **Clips.** 20 short WAVs, all exactly 4.000 seconds at
  16 kHz, sound alone, no picture: 8 tone, 8 silence,
  4 speech. They live in the Drive folder
  `tracing-qwen3b/patching_clips`, and Step 2 copies them
  here. The runner proves every clip against the folder's
  manifest before anything uses it: SHA-256 prefixes,
  seconds, and RMS, printed per clip, hard stop on any
  mismatch.
- **Questions.** The presence questions and the yes score
  are the Series Two runner's own, copied from its code:
  the tone and silence clips are asked "Is there the
  sound of beeping in this clip? Answer yes or no." and
  the speech clips "Is there the sound of a person
  speaking in this clip? Answer yes or no." The yes score
  is the probability of a yes token at the first generated
  position, from the same greedy generate call that
  produces the answer word. The open question, used only
  for a few context answers, is the swap study's P3.
- **Donor and recipient always run under the same
  question**, the direction's scored question, so a patch
  moves sound information, never question information.
- **Baselines first.** Every clip's unpatched yes score
  and answer word, so every patched score has its own
  recipient's starting point beside it.
- **The sweep.** Tone into silence (8 pairs, layers 0 to
  35) and speech into silence (4 pairs, all layers). Each
  patched run is one prefill and a first-token read; its
  row is appended to `patching_results.csv` the moment it
  finishes, so an interruption loses nothing.
- **Three controls.** Silence into tone (does the score
  fall?), tone into a different tone clip (a sham: the
  sound is already there, so little should change), and
  tone into silence patched at the non-audio positions
  only (the patch lands everywhere except the sound's own
  slots; no rescue is expected). A result that cannot
  survive its controls is not a result.
- **The open-form check.** At the single layer where the
  tone patch lifts the silence scores most (over the
  recipients' own baselines), the 8 patched silence runs
  also answer the open question, to test the registered
  prediction that naming is a later step than detection.

**Predictions, registered in the tracing plan before any
of this was built** (quoted exactly in the closing prints;
this notebook prints no verdict on them, the read happens
outside):

- Tone: patching rescues the yes/no answer at mid layers
  but not the open form.
- Speech: patching does NOT fix the denial, because
  nothing is missing: the loss sits in how the presence
  answer itself is formed, past the information.

**Where it runs.** On the free Colab account (kadjei1),
under the routing rule: the 3B is a small model. **How to
run it.** Run the cells from the top, in order. Step 1
sets up and proves the stack. Step 2 gets the patching
clips. Step 3 writes the runner and runs its check mode
(proof, loading candidates, layer resolution, the token
check, the feed proof). Step 4 runs battery mode:
baselines, captures, the sweep, the controls, and the
open-form check.

**The one rule.** We claim only what we ran: these clips,
this model, these layers, this run.
"""))
cells.append(step1_md)
cells.append(step1_code)

# ---------------------------------------------------------------- step 2
cells.append(md(r"""
## Step 2 · Get the patching clips

Twenty WAV files and their manifest, from your Google
Drive folder `tracing-qwen3b/patching_clips`:

- `tone_00.wav` to `tone_07.wav`: the tone clips.
- `silence_00.wav` to `silence_07.wav`: digital silence.
- `speech_00.wav` to `speech_03.wav`: the probe sentence.
- `manifest.json`: each file's SHA-256 prefix and its
  length in seconds, the record the runner's feed proof
  checks the session's copies against.

The next cell links your Drive, finds that folder, and
copies all 21 files to a plain work folder here. The
runner re-measures everything from inside the session
before it believes any of it.

Do not go on until every file says "ready". If a file is
missing, the cell stops and names it.
"""))

cells.append(code(r"""
# --- Get the patching clips -------------------------------------------------------
import os, shutil, subprocess

FILES = (["manifest.json"]
         + [f"tone_{i:02d}.wav" for i in range(8)]
         + [f"silence_{i:02d}.wav" for i in range(8)]
         + [f"speech_{i:02d}.wav" for i in range(4)])
WORK = "/content/patching_clips"
DRIVE_CLIPS = "/content/drive/MyDrive/tracing-qwen3b/patching_clips"
os.makedirs(WORK, exist_ok=True)

try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception as e:
    print("Drive mount skipped:", e)

SOURCES = [DRIVE_CLIPS, "/content/tracing-qwen3b/patching_clips"]

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

missing = []
for name in FILES:
    bring(name)
    path = os.path.join(WORK, name)
    if os.path.exists(path):
        d = duration_s(path)
        extra = f", {d:.3f} s" if d is not None else ""
        print("ready:  ", name, f"({os.path.getsize(path)} bytes{extra})")
    else:
        missing.append(name)
        print("MISSING:", name)
if missing:
    print()
    print("!" * 72)
    print("STOP. These patching files are missing:")
    for name in missing:
        print("   ", name)
    print("Check the Drive folder is tracing-qwen3b/patching_clips, or")
    print("upload the files by hand, then run this cell again.")
    print("Do not go on to Step 3.")
    print("!" * 72)
    raise SystemExit("STOP: patching files missing in Step 2")
print()
print("ready: all 21 patching files (20 clips and the manifest)")
"""))

# ------------------------------------------------------------- runner script
# The runner is a plain Python script the notebook writes to
# /content/qwen_patching_run.py and runs under the notebook
# session's own Python, the Qwen notebooks' own pattern.
RUNNER_SRC = r"""
# qwen_patching_run.py - the surgery (activation patching)
# runner for Qwen2.5-Omni-3B, Series Three test 2. Written
# by project_k_qwen_patching_colab.ipynb. It runs under the
# notebook session's own Python: the Qwen notebooks install
# their stack into the session (Step 1), and there is no
# private environment to run under. Modes: "check" runs the
# proof, the loading candidates, the layer resolution, the
# TOKEN CHECK, and the FEED PROOF, then stops. "battery"
# does all of that and then runs the baselines, the donor
# captures, the patching sweep with its controls, and the
# best-layer open-form check.
#
# Exit codes, one per failure family, all loud:
#   3  the stack proof failed (transformers is not 4.52.4)
#   4  both loading candidates failed
#   7  the feed proof failed (a clip missing, its SHA-256
#      prefix off the manifest, its length off, or the feed
#      path returned no audio or zeros on a sound clip)
#   11 the token check failed: the audio positions could
#      not be settled from the histogram, or their count is
#      not the expected 100
#   12 a patching mechanic failed: the decoder layer list
#      did not resolve to exactly 36 layers, the hidden
#      states disagree with it, or a donor and a recipient
#      row do not line up for a patch
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else "check"
print("runner mode:", MODE)
print("runner python:", sys.version.split()[0], sys.executable)

# --- The fixed seed ------------------------------------------------------------------
# Generation is greedy, so the seed decides nothing about
# the answers; it is fixed and printed anyway, the house
# rule, so every random draw anywhere in the run (there
# should be none that matter) is reproducible.
import random
import numpy as np
import torch

SEED = __SEED__
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
print("fixed seed:", SEED)

# --- Proof lines, printed before anything loads --------------------------------
import transformers
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
print("attention implementation note: the loading candidates below pass")
print("no attn_implementation, so the model loads with the library default")
print("(sdpa), the same door the Series Two asks ran through. This build")
print("reads no attention weights anywhere, so it does not need the eager")
print("mode tracing round 2 had to load for its attention readout.")

# --- Homes for the outputs -------------------------------------------------------
# The patching outputs live in the Drive folder
# tracing-qwen3b beside the tracing rounds' outputs, with
# /content as the second home, the house pattern.
DRIVE_TRACING = "/content/drive/MyDrive/tracing-qwen3b"
LOCAL_TRACING = "/content/tracing-qwen3b"
for d in (DRIVE_TRACING, LOCAL_TRACING):
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass

def out_paths(name):
    return [os.path.join(DRIVE_TRACING, name),
            os.path.join(LOCAL_TRACING, name)]

def save_both(name, write_fn):
    saved = []
    for p in out_paths(name):
        try:
            write_fn(p)
            saved.append(p)
            print("saved:", p)
        except Exception as e:
            print("could not save", p, "-", str(e)[:160])
    return saved

# --- The load log ------------------------------------------------------------------
import traceback, datetime

LOAD_LOG_PATHS = out_paths("patching_load_log.txt")

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
log_load("Qwen2.5-Omni-3B patching (Series Three surgery) load log - "
         + datetime.datetime.now().isoformat(timespec="seconds")
         + " - runner mode: " + MODE + "\n")

# --- Load the model: the tracing runners' candidates ------------------------------
# Same candidates, same order, same talker-off call as the
# tracing runners. The one printed difference is the
# missing attn_implementation (see the proof lines above
# for why).
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

THINKER = model.thinker
print("thinker:", type(THINKER).__name__)
"""

RUNNER_SRC += r"""
# --- Resolve the thinker decoder layers --------------------------------------------
# The patch hooks live on the thinker's decoder layers, so
# the layer list is resolved from the loaded model at
# runtime, never assumed from a version's source layout:
# every module list in the model is printed, the list of
# exactly 36 modules under the thinker is the decoder, and
# anything else stops the run (exit 12) before a single
# hook is set. Sweep layer L means the output of decoder
# layer L, which is hidden_states[L + 1] in a forward's
# returned tuple (index 0 is the embedding output).
N_LAYERS_EXPECTED = __N_LAYERS__
LAYER_PATH = None
LAYERS = None

def mechanics_fail(reason):
    print()
    print("!" * 70)
    print("PATCHING MECHANICS FAILED: " + reason)
    print("No hook has been trusted and no patch score recorded.")
    print("Show this whole output to Muse.")
    print("!" * 70)
    sys.exit(12)

all_lists = [(name, mod) for name, mod in model.named_modules()
             if isinstance(mod, torch.nn.ModuleList)]
print("module lists found in the loaded model:")
for name, mod in all_lists:
    print(f"  {name}: {len(mod)} modules")
exact = [(name, mod) for name, mod in all_lists
         if len(mod) == N_LAYERS_EXPECTED]
thinker_exact = [(name, mod) for name, mod in exact
                 if name.startswith("thinker")]
chosen = thinker_exact if thinker_exact else exact
if len(chosen) != 1:
    mechanics_fail(
        f"the decoder layer list did not resolve to exactly one "
        f"list of {N_LAYERS_EXPECTED} modules (found "
        f"{len(exact)} such lists, {len(thinker_exact)} of them "
        f"under the thinker); the printed module lists above "
        f"are the evidence")
LAYER_PATH, _layers_mod = chosen[0]
LAYERS = list(_layers_mod)
print(f"decoder layer list resolved: {LAYER_PATH}, "
      f"{len(LAYERS)} layers")
print(f"layer module type: {type(LAYERS[0]).__name__}")
log_load("decoder layer list: " + LAYER_PATH + ", "
         + str(len(LAYERS)) + " layers\n")

# --- The yes score: the Series Two runner's own scorer ------------------------------
# Copied from build_series2_qwen_notebook.py, the Qwen
# Series Two runner: the yes token ids are read from the
# model's own tokenizer (the first token of "Yes" and of
# " Yes", lines 830-837 there), and yes_probability (lines
# 995-1010 there) takes the softmax over the vocabulary of
# the first generated position's logits, from the first
# score row of the same greedy generate call that produced
# the answer, summed over those ids. The generate call
# itself (in ask_generate below) is that runner's ask path
# (lines 1171-1178 there): thinker_max_new_tokens,
# do_sample=False, return_dict_in_generate=True,
# output_scores=True, and only the newly written tokens
# are decoded.
YES_TOKEN_IDS = []
for _word in ("Yes", " Yes"):
    _enc = processor.tokenizer.encode(_word, add_special_tokens=False)
    if _enc:
        YES_TOKEN_IDS.append(int(_enc[0]))
YES_TOKEN_IDS = sorted(set(YES_TOKEN_IDS))
print("yes token ids for scoring:", YES_TOKEN_IDS,
      [processor.tokenizer.decode([i]) for i in YES_TOKEN_IDS])

def yes_probability(scores):
    if not scores:
        return None
    logits = scores[0][0].float()
    probs = torch.softmax(logits, dim=-1)
    total = 0.0
    for i in YES_TOKEN_IDS:
        total += float(probs[i].item())
    return total

# --- The feed path -----------------------------------------------------------------
# Sound alone, the tracing runners' feed form: the
# conversation carries an audio element naming the WAV file
# and a text element, and no image and no video element.
# process_mm_info decodes the audio element with the same
# librosa call at 16 kHz mono it uses for a video
# soundtrack.
import json, math
import librosa
import soundfile as sf
from qwen_omni_utils import process_mm_info

PATCH_CLIPS = "/content/patching_clips"
PROMPT_BEEPS = __PROMPT_BEEPS__
PROMPT_SPEECH = __PROMPT_SPEECH__
PROMPT_OPEN = __P3__
QUESTION_FOR = {"tone": PROMPT_BEEPS, "silence": PROMPT_BEEPS,
                "speech": PROMPT_SPEECH}
TARGET_FOR = {"tone": __TARGET_BEEPS__, "silence": __TARGET_BEEPS__,
              "speech": __TARGET_SPEECH__}

class FeedError(RuntimeError):
    pass

def conversation_for(path, prompt):
    return [{
        "role": "user",
        "content": [
            {"type": "audio", "audio": path},
            {"type": "text", "text": prompt},
        ],
    }]

def feed_inputs(path, prompt, expect_silence=False):
    # Run the feed path for one clip and build the model's
    # inputs. Raises FeedError if the path returns no audio:
    # a missing track stops the run, it is never padded. A
    # track that decodes to all zeros also stops the run,
    # unless the caller declares the clip silent by design
    # (a patching silence clip): an all-zeros feed on a clip
    # that should carry sound is a broken feed, not a quiet
    # clip.
    if not os.path.exists(path):
        raise FeedError(f"feed file missing: {path}")
    conversation = conversation_for(path, prompt)
    audios, images, videos = process_mm_info(
        conversation, use_audio_in_video=False)
    if audios is None or len(audios) == 0 or audios[0] is None:
        raise FeedError(
            f"the feed path returned no audio for {path}")
    audio_np = np.asarray(audios[0], dtype=np.float32)
    if len(audio_np) == 0:
        raise FeedError(
            f"decoded track is empty for {path}")
    if not expect_silence and not np.any(audio_np != 0):
        raise FeedError(
            f"decoded track is all zeros for {path}, on a clip "
            f"that should carry sound")
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audio=audios,
                       return_tensors="pt", padding=True)
    inputs = inputs.to(model.device).to(model.dtype)
    return inputs, audio_np

def thinker_forward(inputs):
    # One frozen pass through the thinker with every layer's
    # numbers saved. The model is never trained; this is a
    # reading, not a conversation.
    with torch.no_grad():
        out = THINKER(**inputs, output_hidden_states=True,
                      output_attentions=False)
    return out
"""

RUNNER_SRC += r"""
# --- The token check -----------------------------------------------------------------
# The tracing round 2 design, with the assert the brief for
# this build fixes: for one tone clip and one speech clip
# from the patching set, print the FULL histogram of token
# ids in the processed input row (every id with count > 0,
# its token name and decoded form, special tokens flagged),
# print the processor's audio feature lengths, count every
# candidate audio id in the row, and DERIVE the audio token
# id from the histogram: the candidate with the highest
# count, the one that occurs at audio-frame scale. Both
# clips must derive the same id, and its count must be
# exactly EXPECTED_AUDIO_POSITIONS (100 for these 4.0 s
# clips), or the step prints TOKEN CHECK FAILED and stops
# the run (exit 11) before any baseline or patch. The audio
# only rows here are shorter than the tracing rounds' rows
# (no picture tokens); nothing about the totals is assumed.
# One thinker forward on the tone clip then confirms the
# hidden states come back and cross-checks the layer count
# against the resolved decoder list. Every line is saved
# to patching_token_check.txt, pass or fail.
from collections import Counter

EXPECTED_AUDIO_POSITIONS = __EXPECTED_AUDIO_POSITIONS__
AUDIO_TOKEN_ID = None   # set by the token check, from the histogram
N_STATES = None
HIDDEN_SIZE = None
TOKEN_CHECK_LINES = []

def tc(line=""):
    print(line)
    TOKEN_CHECK_LINES.append(line)

def save_token_check():
    text = "\n".join(TOKEN_CHECK_LINES) + "\n"
    def w(p):
        with open(p, "w") as fh:
            fh.write(text)
    save_both("patching_token_check.txt", w)

def token_check_fail(reason):
    tc()
    tc("!" * 70)
    tc("TOKEN CHECK FAILED: " + reason)
    tc("The audio positions are not settled, so no baseline")
    tc("has run and no patch score in this run can be trusted.")
    tc("The full histogram above is the evidence. Show this whole")
    tc("output to Muse.")
    tc("!" * 70)
    save_token_check()
    sys.exit(11)

def audio_candidates():
    # The candidate audio ids, from the model's own
    # tokenizer and config, never a guessed number. Same
    # evidence the tracing checks gathered; the winner is
    # picked from the histogram (see token_check).
    tok = processor.tokenizer
    vocab = tok.get_vocab()
    audio_named = {t: i for t, i in vocab.items() if "audio" in t}
    tc("tokenizer tokens whose text mentions audio:")
    for t in sorted(audio_named):
        tc(f"  {t!r} -> id {audio_named[t]}")
    candidates = {}
    pad_id = tok.convert_tokens_to_ids("<|audio_pad|>")
    if pad_id is not None and int(pad_id) >= 0:
        candidates[int(pad_id)] = "<|audio_pad|> (converted by name)"
    for t, i in audio_named.items():
        candidates[int(i)] = f"tokenizer token {t!r}"
    for holder_name, holder in (
            ("model.config", model.config),
            ("thinker_config",
             getattr(model.config, "thinker_config", None))):
        if holder is None:
            continue
        for attr in ("audio_token_id", "audio_token_index"):
            val = getattr(holder, attr, None)
            if val is not None:
                tc(f"  {holder_name}.{attr} = {val}")
                candidates.setdefault(
                    int(val), f"config {holder_name}.{attr}")
    return candidates

def histogram_for(label, path, prompt, candidates):
    # Feed one patching clip and print its full token
    # histogram plus the processor's audio feature lengths.
    # Returns (inputs, chosen_id, chosen_count, row_length).
    tok = processor.tokenizer
    try:
        inputs, audio_np = feed_inputs(path, prompt)
    except FeedError as e:
        token_check_fail(
            f"the {label} clip's feed failed inside the token "
            f"check: {e}")
    row = [int(x) for x in inputs.input_ids[0].tolist()]
    tc("-" * 70)
    tc(f"TOKEN HISTOGRAM: {label} clip")
    tc(f"  file: {path}")
    tc(f"  decoded waveform: {len(audio_np)} samples at 16 kHz "
       f"({len(audio_np) / 16000:.2f} s)")
    tc(f"  sequence length: {len(row)} positions")
    counts = Counter(row)
    tc(f"  distinct token ids: {len(counts)}")
    tc("  full histogram, most common first (every id with "
       "count > 0):")
    special_ids = set(int(x) for x in tok.all_special_ids)
    for tid, cnt in counts.most_common():
        name = tok.convert_ids_to_tokens([tid])[0]
        decoded = tok.decode([tid])
        flag = " [special token]" if tid in special_ids else ""
        tc(f"    id {tid}: count {cnt}, token name {name!r}, "
           f"decodes to {decoded!r}{flag}")
    feats = getattr(inputs, "input_features", None)
    if feats is not None:
        tc(f"  processor input_features shape: "
           f"{tuple(feats.shape)} (batch, mel bins, audio frames)")
    else:
        tc("  processor input_features: not present on the inputs")
    fam = getattr(inputs, "feature_attention_mask", None)
    if fam is not None:
        tc(f"  processor feature_attention_mask shape: "
           f"{tuple(fam.shape)}, true frames per row: "
           f"{[int(x) for x in fam.sum(dim=-1).tolist()]}")
    else:
        tc("  processor feature_attention_mask: not present")
    tc("  candidate audio ids, counted in this row:")
    cand_counts = {}
    for cid, why in sorted(candidates.items()):
        cnt = row.count(cid)
        cand_counts[cid] = cnt
        tc(f"    candidate id {cid} ({why}): count {cnt}, "
           f"decodes to {tok.decode([cid])!r}")
    chosen = max(cand_counts, key=lambda c: cand_counts[c])
    positions = [i for i, x in enumerate(row) if x == chosen]
    tc(f"  DERIVED audio token id for the {label} clip: {chosen} "
       f"({candidates[chosen]}), decodes to "
       f"{tok.decode([chosen])!r}")
    tc(f"  audio positions: {len(positions)} of {len(row)}; "
       f"first {positions[:5]}, last {positions[-5:]}")
    return inputs, chosen, cand_counts[chosen], len(row)

def token_check():
    global AUDIO_TOKEN_ID, N_STATES, HIDDEN_SIZE
    tc("=" * 70)
    tc("TOKEN CHECK: settle the audio positions from the "
       "histogram, before any baseline or patch")
    tc(f"expected audio-position count for these 4.0 s clips: "
       f"{EXPECTED_AUDIO_POSITIONS}")
    candidates = audio_candidates()
    tone_path = os.path.join(PATCH_CLIPS, "tone_00.wav")
    speech_path = os.path.join(PATCH_CLIPS, "speech_00.wav")
    tone_inputs, tone_id, tone_count, tone_len = histogram_for(
        "tone", tone_path, PROMPT_BEEPS, candidates)
    _, speech_id, speech_count, speech_len = histogram_for(
        "speech", speech_path, PROMPT_SPEECH, candidates)
    if tone_id != speech_id:
        token_check_fail(
            f"the two clips derive different audio token ids "
            f"(tone {tone_id}, speech {speech_id}); the audio "
            f"positions cannot be settled on a split answer")
    if tone_count != EXPECTED_AUDIO_POSITIONS \
            or speech_count != EXPECTED_AUDIO_POSITIONS:
        token_check_fail(
            f"the derived audio-position count is not the "
            f"expected {EXPECTED_AUDIO_POSITIONS} (tone "
            f"{tone_count}, speech {speech_count}); patching "
            f"replaces every audio position, so a count that "
            f"differs from the design stops the run")
    AUDIO_TOKEN_ID = tone_id
    tc("-" * 70)
    tc(f"AUDIO TOKEN ID SETTLED: {AUDIO_TOKEN_ID}, decodes to "
       f"{processor.tokenizer.decode([AUDIO_TOKEN_ID])!r}")
    tc(f"audio positions: tone {tone_count} of {tone_len} row "
       f"positions; speech {speech_count} of {speech_len}")
    out = thinker_forward(tone_inputs)
    hs = getattr(out, "hidden_states", None)
    if hs is None or len(hs) == 0:
        token_check_fail(
            "the thinker returned no hidden states on the tone "
            "clip; there would be nothing to capture or patch")
    N_STATES = len(hs)
    HIDDEN_SIZE = int(hs[0].shape[2])
    tc(f"hidden states returned: {N_STATES} tensors (index 0 is "
       f"the embedding output, then one per decoder layer)")
    tc(f"hidden size: {HIDDEN_SIZE}")
    if N_STATES - 1 != len(LAYERS):
        tc(f"hidden states imply {N_STATES - 1} decoder layers "
           f"but the resolved layer list holds {len(LAYERS)}")
        save_token_check()
        mechanics_fail(
            "the hidden states and the resolved decoder layer "
            "list disagree on the layer count")
    tc("TOKEN CHECK PASSED: the audio positions are derived "
       "from the histogram, counted, and the same for both "
       "clips. Every capture and patch in this run works on "
       "these positions.")
    save_token_check()
    del out, tone_inputs
    free_memory()

token_check()

def audio_positions(input_ids_row):
    row = [int(x) for x in input_ids_row]
    return [i for i, x in enumerate(row) if x == AUDIO_TOKEN_ID]
"""

RUNNER_SRC += r"""
# --- The feed proof ------------------------------------------------------------------
# Every clip the run uses is proven against the clips'
# manifest before any baseline or patch: the file's own
# SHA-256 prefix must match the manifest's sha256_16, the
# decoded length must match the manifest's seconds, and the
# sample rate and RMS are printed per clip. The manifest's
# hashes are of the file bytes (verified against the files
# when this notebook was built). Any mismatch prints FEED
# PROOF FAILED and stops the run (exit 7): nothing is ever
# scored on an unproven clip.
import csv, hashlib

def rms_db_np(x):
    if len(x) == 0:
        return -120.0
    v = float(np.sqrt(np.mean(np.square(x.astype(np.float64)))))
    return 20.0 * math.log10(v) if v > 0 else -120.0

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

CLIP_INFO = {}   # clip name (no .wav) -> record

def feed_proof_fail(reason):
    print()
    print("!" * 70)
    print("FEED PROOF FAILED: " + reason)
    print("No baseline has been taken and no patch scored on an")
    print("unproven clip. Show this whole output to Muse.")
    print("!" * 70)
    sys.exit(7)

def feed_proof():
    manifest_path = os.path.join(PATCH_CLIPS, "manifest.json")
    if not os.path.exists(manifest_path):
        feed_proof_fail(f"the clips' manifest is missing: "
                        f"{manifest_path}")
    with open(manifest_path) as fh:
        manifest = json.load(fh)
    print("=" * 70)
    print(f"FEED PROOF: {len(manifest)} clips against the manifest")
    for entry in manifest:
        name = entry["file"]
        clip = name[:-4] if name.endswith(".wav") else name
        group = clip.split("_")[0]
        path = os.path.join(PATCH_CLIPS, name)
        if not os.path.exists(path):
            feed_proof_fail(f"clip file missing: {path}")
        digest = file_sha256(path)
        if not digest.startswith(entry["sha256_16"]):
            feed_proof_fail(
                f"{name}: the file's SHA-256 begins "
                f"{digest[:16]}, the manifest says "
                f"{entry['sha256_16']}; this is not the clip "
                f"the manifest describes")
        wave, sr = sf.read(path, dtype="float32")
        if wave.ndim > 1:
            wave = wave[:, 0]
        seconds = len(wave) / sr if sr else 0.0
        if abs(seconds - float(entry["seconds"])) > 0.001:
            feed_proof_fail(
                f"{name}: decoded length {seconds:.3f} s, the "
                f"manifest says {entry['seconds']} s")
        rms = rms_db_np(wave)
        CLIP_INFO[clip] = {"clip": clip, "group": group,
                           "file": name, "path": path,
                           "seconds": seconds, "sr": int(sr),
                           "rms_db": round(rms, 2)}
        print(f"  feed proof {name}: {seconds:.3f} s, "
              f"{len(wave)} samples at {sr} Hz, RMS {rms:.2f} dB, "
              f"sha256 prefix {digest[:16]} matches the manifest")
    print(f"feed proof passed: {len(CLIP_INFO)} clips proven "
          f"against the manifest")

feed_proof()

if MODE == "check":
    print()
    print("check mode done: proof, candidates, layer resolution,")
    print("the token check, and the feed proof all passed. No")
    print("baselines were taken and no patch was run.")
    print("Run Step 4 for the battery.")
    sys.exit(0)
"""

RUNNER_SRC += r"""
# --- Ask machinery ---------------------------------------------------------------------
# One generate call shape for every answer in this build,
# the Series Two ask path's audio alone arm: the processor
# builds the inputs from the conversation, and the wrapper
# generate runs greedy with the cap passed as
# thinker_max_new_tokens, returning its scores so a yes/no
# ask records the yes probability from the first score row
# of the same call that produced the answer. Only the newly
# written tokens are decoded.
PRESENCE_MAX_TOKENS = __PRESENCE_TOKENS__
OPEN_MAX_TOKENS = __OPEN_TOKENS__
RESULTS_FIELDS = __RESULTS_FIELDS__
BASELINE_FIELDS = __BASELINE_FIELDS__
OPENFORM_FIELDS = __OPENFORM_FIELDS__

def build_inputs(clip, prompt):
    info = CLIP_INFO[clip]
    try:
        inputs, _ = feed_inputs(
            info["path"], prompt,
            expect_silence=(info["group"] == "silence"))
    except FeedError as e:
        print()
        print("!" * 70)
        print("FEED HARD STOP:")
        print(str(e))
        print(f"Clip {clip} was not run, and the run stops here.")
        print("No padding was substituted. Show this whole output")
        print("to Muse.")
        print("!" * 70)
        sys.exit(7)
    return inputs

def generate_from_inputs(inputs, max_tokens, want_yes):
    with torch.no_grad():
        out = model.generate(
            **inputs, thinker_max_new_tokens=max_tokens,
            do_sample=False, return_dict_in_generate=True,
            output_scores=True)
    new_tokens = out.sequences[:, inputs.input_ids.shape[1]:]
    decoded = processor.batch_decode(
        new_tokens, skip_special_tokens=True,
        clean_up_tokenization_spaces=False)[0].strip()
    yp = yes_probability(out.scores) if want_yes else None
    del out, new_tokens
    return decoded, yp

def ask_unpatched(clip, prompt, max_tokens, want_yes):
    inputs = build_inputs(clip, prompt)
    decoded, yp = generate_from_inputs(inputs, max_tokens, want_yes)
    del inputs
    free_memory()
    return decoded, yp

# --- Baselines ---------------------------------------------------------------------------
# Every clip's unpatched presence score and answer word
# under its own question (tone and silence under the
# beeping question, speech under the speaking question),
# then the open question on three clips as context. Saved
# whole to patching_baselines.csv; every later patched
# score is read against its own recipient's baseline.
print("=" * 70)
print("BASELINES: unpatched, every clip")
baseline_rows = []
baseline_yes = {}
for clip in (__TONE_CLIPS__ + __SILENCE_CLIPS__ + __SPEECH_CLIPS__):
    prompt = QUESTION_FOR[CLIP_INFO[clip]["group"]]
    decoded, yp = ask_unpatched(clip, prompt, PRESENCE_MAX_TOKENS,
                                True)
    baseline_yes[clip] = yp
    baseline_rows.append({
        "clip": clip, "question_target": TARGET_FOR[
            CLIP_INFO[clip]["group"]],
        "prompt": prompt, "form": "presence",
        "yes_score": f"{yp:.4f}", "answer": decoded})
    print(f"  baseline {clip}: yes {yp:.4f}, answer {decoded!r}")
for clip in ("tone_00", "silence_00", "speech_00"):
    decoded, _ = ask_unpatched(clip, PROMPT_OPEN, OPEN_MAX_TOKENS,
                                False)
    baseline_rows.append({
        "clip": clip, "question_target": "", "prompt": PROMPT_OPEN,
        "form": "open_p3", "yes_score": "", "answer": decoded})
    print(f"  baseline open {clip}: {decoded!r}")

def write_baselines(p):
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=BASELINE_FIELDS)
        w.writeheader()
        w.writerows(baseline_rows)
save_both("patching_baselines.csv", write_baselines)

# --- Donor captures ------------------------------------------------------------------------
# A capture is one frozen thinker forward on a donor clip
# under the direction's question: the full hidden states
# (every layer's output, index 0 the embedding output),
# kept on the CPU as float16, with the input row and its
# audio positions. Patching at sweep layer L uses index
# L + 1, the output of decoder layer L. Captures are taken
# once per donor per question and reused for every layer
# and every recipient in that direction.
print("=" * 70)
print("CAPTURES: donor runs, one per donor per question")

def capture(clip, prompt):
    inputs = build_inputs(clip, prompt)
    row = [int(x) for x in inputs.input_ids[0].tolist()]
    pos = audio_positions(row)
    if len(pos) != EXPECTED_AUDIO_POSITIONS:
        mechanics_fail(
            f"donor {clip} has {len(pos)} audio positions, not "
            f"the {EXPECTED_AUDIO_POSITIONS} the token check "
            f"settled; a capture on unsettled positions cannot "
            f"be patched from")
    out = thinker_forward(inputs)
    hs = getattr(out, "hidden_states", None)
    if hs is None or len(hs) != len(LAYERS) + 1:
        mechanics_fail(
            f"donor {clip}: the thinker returned "
            f"{0 if hs is None else len(hs)} hidden state "
            f"tensors, not the {len(LAYERS) + 1} the layer list "
            f"implies")
    states = [hs[k][0].detach().to("cpu", torch.float16)
              for k in range(len(hs))]
    del out, inputs
    free_memory()
    print(f"  captured {clip} under the "
          f"{TARGET_FOR[CLIP_INFO[clip]['group']]} question: "
          f"row {len(row)} positions, {len(pos)} audio, "
          f"{len(states)} state tensors")
    return {"clip": clip, "row": row, "positions": pos,
            "states": states}

CAPTURES = {}
for clip in __TONE_CLIPS__:
    CAPTURES[(clip, PROMPT_BEEPS)] = capture(clip, PROMPT_BEEPS)
for clip in __SILENCE_CLIPS__[:4]:
    CAPTURES[(clip, PROMPT_BEEPS)] = capture(clip, PROMPT_BEEPS)
for clip in __SPEECH_CLIPS__:
    CAPTURES[(clip, PROMPT_SPEECH)] = capture(clip, PROMPT_SPEECH)
print(f"captures held: {len(CAPTURES)} donor runs")
"""

RUNNER_SRC += r"""
# --- The patch --------------------------------------------------------------------------------
# One patched run: build the recipient's inputs, set a
# forward hook on decoder layer L, and generate. The hook
# fires on every forward through that layer; it patches
# only the full length forward (the prefill that decides
# the first generated token) and leaves the one token
# decode steps alone, guarded by the sequence length. The
# patch overwrites, at the chosen positions, the layer's
# output with the donor's output at the same layer, donor
# position by donor position in order. The hook is always
# removed, answer or error.
def patched_generate(recipient, prompt, donor_cap, layer, mode,
                     max_tokens, want_yes):
    inputs = build_inputs(recipient, prompt)
    row = [int(x) for x in inputs.input_ids[0].tolist()]
    if len(row) != len(donor_cap["row"]):
        mechanics_fail(
            f"recipient {recipient} has a {len(row)} position "
            f"row but donor {donor_cap['clip']} has "
            f"{len(donor_cap['row'])}; a patch lines positions "
            f"up one to one, so these two runs cannot be paired")
    recip_audio = audio_positions(row)
    if len(recip_audio) != EXPECTED_AUDIO_POSITIONS:
        mechanics_fail(
            f"recipient {recipient} has {len(recip_audio)} "
            f"audio positions, not {EXPECTED_AUDIO_POSITIONS}")
    if mode == "audio":
        recip_pos = recip_audio
        donor_pos = donor_cap["positions"]
    elif mode == "nonaudio":
        dset = set(donor_cap["positions"])
        donor_pos = [i for i in range(len(donor_cap["row"]))
                     if i not in dset]
        rset = set(recip_audio)
        recip_pos = [i for i in range(len(row)) if i not in rset]
    else:
        mechanics_fail(f"unknown patch mode {mode!r}")
    if len(recip_pos) != len(donor_pos):
        mechanics_fail(
            f"recipient {recipient} offers {len(recip_pos)} "
            f"{mode} positions but donor {donor_cap['clip']} "
            f"offers {len(donor_pos)}; the patch cannot line up")
    donor_vec = donor_cap["states"][layer + 1][donor_pos]
    full_len = len(row)
    pos_index = torch.tensor(recip_pos, dtype=torch.long)
    target_layer = LAYERS[layer]

    def hook(module, args, output):
        is_tuple = isinstance(output, tuple)
        is_list = isinstance(output, list)
        hs = output[0] if (is_tuple or is_list) else output
        if hs.shape[1] != full_len:
            return output   # a decode step, not the prefill
        new = hs.clone()
        new[0, pos_index.to(hs.device), :] = donor_vec.to(
            device=hs.device, dtype=hs.dtype)
        if is_tuple:
            return (new,) + output[1:]
        if is_list:
            return [new] + list(output[1:])
        return new

    handle = target_layer.register_forward_hook(hook)
    try:
        decoded, yp = generate_from_inputs(inputs, max_tokens,
                                           want_yes)
    finally:
        handle.remove()
    del inputs
    free_memory()
    return decoded, yp

# --- The sweep, checkpointed ---------------------------------------------------------------------
# Rows are appended to patching_results.csv in both homes
# the moment each patched run finishes, so an interruption
# loses at most the run in flight. A fresh battery starts
# clean: an existing results file is first renamed aside
# with a timestamp, never silently overwritten and never
# duplicated into.
RESULTS_NAME = "patching_results.csv"
for p in out_paths(RESULTS_NAME):
    if os.path.exists(p) and os.path.getsize(p) > 0:
        aside = p + ".partial-" + datetime.datetime.now().strftime(
            "%Y%m%d-%H%M%S")
        try:
            os.rename(p, aside)
            print(f"note: an earlier {RESULTS_NAME} was moved "
                  f"aside to {aside}; this battery starts clean")
        except Exception as e:
            print(f"could not move {p} aside ({str(e)[:120]}); "
                  f"appends would mix runs, so the run stops")
            sys.exit(12)

append_warned = set()

def append_results(new_rows):
    for p in out_paths(RESULTS_NAME):
        try:
            exists = os.path.exists(p) and os.path.getsize(p) > 0
            with open(p, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=RESULTS_FIELDS)
                if not exists:
                    w.writeheader()
                w.writerows(new_rows)
        except Exception as e:
            if p not in append_warned:
                append_warned.add(p)
                print(f"WARNING: could not append results to "
                      f"{p} ({str(e)[:120]}); the other home "
                      f"still receives every row")

results_count = 0
tone_lifts = {}   # layer -> list of (patched yes - recipient baseline)

def run_sweep(direction, pairs, layers, prompt, mode="audio"):
    global results_count
    print("=" * 70)
    print(f"SWEEP: {direction}, {len(pairs)} pairs x "
          f"{len(layers)} layers, mode {mode}")
    for donor, recipient in pairs:
        donor_cap = CAPTURES[(donor, prompt)]
        for layer in layers:
            decoded, yp = patched_generate(
                recipient, prompt, donor_cap, layer, mode,
                PRESENCE_MAX_TOKENS, True)
            row_out = {
                "pair": f"{donor}>{recipient}", "donor": donor,
                "recipient": recipient, "direction": direction,
                "layer": layer, "yes_score": f"{yp:.4f}",
                "answer_word": decoded}
            append_results([row_out])
            results_count += 1
            print(f"  {direction} {donor}>{recipient} layer "
                  f"{layer}: yes {yp:.4f}, answer {decoded!r}")
            if direction == "tone_to_silence":
                lift = yp - baseline_yes[recipient]
                tone_lifts.setdefault(layer, []).append(lift)
    print(f"  {direction} done; results rows so far: "
          f"{results_count}")

ALL_LAYERS = list(range(len(LAYERS)))
run_sweep("tone_to_silence",
          [(t, s) for t, s in zip(__TONE_CLIPS__,
                                  __SILENCE_CLIPS__)],
          ALL_LAYERS, PROMPT_BEEPS)
run_sweep("speech_to_silence",
          [(sp, sil) for sp, sil in zip(__SPEECH_CLIPS__,
                                        __SILENCE_CLIPS__)],
          ALL_LAYERS, PROMPT_SPEECH)
run_sweep("silence_to_tone",
          [(sil, t) for sil, t in zip(__SILENCE_CLIPS__,
                                      __TONE_CLIPS__)][:4],
          ALL_LAYERS, PROMPT_BEEPS)
run_sweep("tone_to_tone",
          [(t, __TONE_CLIPS__[(i + 1) % len(__TONE_CLIPS__)])
           for i, t in enumerate(__TONE_CLIPS__[:4])],
          ALL_LAYERS, PROMPT_BEEPS)
run_sweep("wrong_positions",
          [(t, s) for t, s in zip(__TONE_CLIPS__,
                                  __SILENCE_CLIPS__)][:4],
          __WRONG_LAYERS__, PROMPT_BEEPS, mode="nonaudio")
print(f"sweep complete: {results_count} patched runs recorded "
      f"(the design's count is {__EXPECTED_ROWS__})")
if results_count != __EXPECTED_ROWS__:
    print("WARNING: the recorded row count differs from the "
          "design's count; the CSV is the record, and this "
          "warning is part of it")

# --- The best-layer open-form check ---------------------------------------------------------------
# The layer where the tone patch lifted the silence
# recipients' yes scores most, on average over the 8 pairs,
# against each recipient's own unpatched baseline. At that
# one layer, the 8 patched silence runs answer the open
# question: does the name arrive with the detection, or
# not? Saved to patching_openform.csv.
print("=" * 70)
print("BEST LAYER: mean tone_to_silence lift per layer, over "
      "the recipients' baselines")
mean_lifts = {}
for layer in ALL_LAYERS:
    lifts = tone_lifts.get(layer, [])
    mean_lifts[layer] = (sum(lifts) / len(lifts)) if lifts \
        else float("nan")
    print(f"  layer {layer}: mean lift "
          f"{mean_lifts[layer]:+.4f} over {len(lifts)} pairs")
best_layer = max(ALL_LAYERS, key=lambda l: mean_lifts[l])
print(f"best layer: {best_layer} (mean lift "
      f"{mean_lifts[best_layer]:+.4f})")
open_rows = []
for tone in __TONE_CLIPS__:
    if (tone, PROMPT_OPEN) not in CAPTURES:
        CAPTURES[(tone, PROMPT_OPEN)] = capture(tone, PROMPT_OPEN)
for tone, silence in zip(__TONE_CLIPS__, __SILENCE_CLIPS__):
    decoded, _ = patched_generate(
        silence, PROMPT_OPEN, CAPTURES[(tone, PROMPT_OPEN)],
        best_layer, "audio", OPEN_MAX_TOKENS, False)
    open_rows.append({"pair": f"{tone}>{silence}",
                      "layer": best_layer, "answer": decoded})
    print(f"  open patched {tone}>{silence} at layer "
          f"{best_layer}: {decoded!r}")

def write_openform(p):
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=OPENFORM_FIELDS)
        w.writeheader()
        w.writerows(open_rows)
save_both("patching_openform.csv", write_openform)
unpatched_silence_open = next(
    (r["answer"] for r in baseline_rows
     if r["clip"] == "silence_00" and r["form"] == "open_p3"),
    "")
print(f"for comparison, the unpatched silence_00 open answer "
      f"from the baselines: {unpatched_silence_open!r}")

# --- Closing prints -----------------------------------------------------------------------------------
print()
print("=" * 70)
print("patching done")
print(f"model: {MODEL_NAME} in {LOAD_MODE} form; layers patched: "
      f"{LAYER_PATH}, {len(LAYERS)} layers; audio token id "
      f"{AUDIO_TOKEN_ID}, {EXPECTED_AUDIO_POSITIONS} positions "
      f"per row")
print("Pre-registered predictions from the tracing plan, quoted")
print("exactly. This notebook prints no verdict on them; the")
print("read happens outside:")
print("  Tone: \"Patching rescues the yes/no answer at mid "
      "layers but not the open form (naming is a later step "
      "than detection).\"")
print("  Speech: \"Patching does NOT fix the denial, because "
      "nothing is missing: the loss sits in how the presence "
      "answer itself is formed, past the information.\"")
print("Outputs: patching_load_log.txt, patching_token_check.txt,")
print("patching_baselines.csv, patching_results.csv,")
print("patching_openform.csv.")
"""

RUNNER_SRC = RUNNER_SRC.replace("__SEED__", str(SEED))
RUNNER_SRC = RUNNER_SRC.replace("__N_LAYERS__", str(N_LAYERS_EXPECTED))
RUNNER_SRC = RUNNER_SRC.replace("__EXPECTED_AUDIO_POSITIONS__",
                                str(EXPECTED_AUDIO_POSITIONS))
RUNNER_SRC = RUNNER_SRC.replace("__PROMPT_BEEPS__", repr(PROMPT_BEEPS))
RUNNER_SRC = RUNNER_SRC.replace("__PROMPT_SPEECH__", repr(PROMPT_SPEECH))
RUNNER_SRC = RUNNER_SRC.replace("__P3__", repr(P3))
RUNNER_SRC = RUNNER_SRC.replace("__TARGET_BEEPS__", repr(TARGET_BEEPS))
RUNNER_SRC = RUNNER_SRC.replace("__TARGET_SPEECH__", repr(TARGET_SPEECH))
RUNNER_SRC = RUNNER_SRC.replace("__PRESENCE_TOKENS__",
                                str(PRESENCE_MAX_TOKENS))
RUNNER_SRC = RUNNER_SRC.replace("__OPEN_TOKENS__", str(OPEN_MAX_TOKENS))
RUNNER_SRC = RUNNER_SRC.replace("__RESULTS_FIELDS__",
                                repr(RESULTS_FIELDS))
RUNNER_SRC = RUNNER_SRC.replace("__BASELINE_FIELDS__",
                                repr(BASELINE_FIELDS))
RUNNER_SRC = RUNNER_SRC.replace("__OPENFORM_FIELDS__",
                                repr(OPENFORM_FIELDS))
RUNNER_SRC = RUNNER_SRC.replace("__TONE_CLIPS__", repr(TONE_CLIPS))
RUNNER_SRC = RUNNER_SRC.replace("__SILENCE_CLIPS__", repr(SILENCE_CLIPS))
RUNNER_SRC = RUNNER_SRC.replace("__SPEECH_CLIPS__", repr(SPEECH_CLIPS))
RUNNER_SRC = RUNNER_SRC.replace("__WRONG_LAYERS__",
                                repr(WRONG_POSITION_LAYERS))
RUNNER_SRC = RUNNER_SRC.replace("__EXPECTED_ROWS__",
                                str(EXPECTED_RESULTS_ROWS))
for _ph in ("__SEED__", "__N_LAYERS__", "__EXPECTED_AUDIO_POSITIONS__",
            "__PROMPT_BEEPS__", "__PROMPT_SPEECH__", "__P3__",
            "__TARGET_BEEPS__", "__TARGET_SPEECH__",
            "__PRESENCE_TOKENS__", "__OPEN_TOKENS__",
            "__RESULTS_FIELDS__", "__BASELINE_FIELDS__",
            "__OPENFORM_FIELDS__", "__TONE_CLIPS__",
            "__SILENCE_CLIPS__", "__SPEECH_CLIPS__",
            "__WRONG_LAYERS__", "__EXPECTED_ROWS__"):
    assert _ph not in RUNNER_SRC, f"placeholder survived: {_ph}"

# ---------------------------------------------------------------- step 3
cells.append(md(r"""
## Step 3 · Write the runner, then check: proof, candidates, layers, token check, feed proof

The model never runs inside this notebook's cells directly.
It runs in a plain script, `/content/qwen_patching_run.py`,
under this session's own Python, the Qwen notebooks'
pattern. The first cell below writes that script. The
second runs it in **check mode**.

Check mode does five things, in order:

1. **Proof again.** The versions and files the runner
   imported, with the transformers pin checked a second
   time: exactly 4.52.4, or the run stops. One note
   explains this build's single deliberate difference from
   the tracing runners' load call: no eager attention,
   because nothing here reads attention weights, so the
   model loads with the library default, the door the
   Series Two asks ran through.
2. **The loading candidates.** The tracing runners' list:
   the 3B in 16-bit form (float16) first, the 7B AWQ build
   second, only if the small one fails. The talker is
   turned off on whichever loads. A failed candidate
   prints its full traceback and appends the same words to
   `patching_load_log.txt` in the tracing folder.
3. **Layer resolution.** Every module list in the loaded
   model is printed, and the decoder is the list of
   exactly 36 modules under the thinker. The resolved path
   and count print; anything else stops the run before a
   single hook is set. This is resolved from the loaded
   model, never assumed from a version's layout.
4. **The token check.** One tone clip and one speech clip
   from the patching set go through the processor. For
   each, the runner prints the full histogram of token ids
   in the processed row (every id, its count, its decoded
   name, special tokens flagged) and the processor's audio
   feature lengths. The audio token id is derived from the
   histogram: the candidate that occurs at audio-frame
   scale. Both clips must derive the same id, and its
   count must be exactly 100, the count these 4.0 second
   clips are built to produce, or the step prints TOKEN
   CHECK FAILED and stops the run (exit code 11) before
   any baseline. One forward pass then cross-checks the
   hidden states' layer count against the resolved list.
   The whole transcript is saved as
   `patching_token_check.txt`.
5. **The feed proof.** Every one of the 20 clips is
   checked against `manifest.json`: the file's SHA-256
   prefix, its decoded length in seconds, its sample rate
   and RMS, printed per clip. Any mismatch stops the run
   (exit code 7). Nothing is ever scored on an unproven
   clip.

No baselines are taken and no patch is run in check mode.
"""))

cells.append(code(
    "# --- Write the patching runner script --------------------------------------------\n"
    "# The runner holds the proof, the loading candidates, the\n"
    "# layer resolution, the token check, the feed proof, the\n"
    "# baselines, the captures, the patching sweep, and the\n"
    "# open-form check. It is written to a plain file here and\n"
    "# only ever run under this session's own Python, the\n"
    "# interpreter Step 1 installed into.\n"
    "RUNNER_SOURCE = " + repr(RUNNER_SRC) + "\n"
    "with open(\"/content/qwen_patching_run.py\", \"w\") as fh:\n"
    "    fh.write(RUNNER_SOURCE)\n"
    "print(\"runner written:\", \"/content/qwen_patching_run.py\", "
    "len(RUNNER_SOURCE), \"chars\")\n"))

cells.append(code(r"""
# --- Check mode: proof, candidates, layers, token check, feed proof -------------------
# Runs the runner under this session's own Python and streams
# its output here live. A nonzero exit code means the proof, a
# loading candidate, the layer resolution, the token check, or
# the feed proof failed, and the run stops in this cell. The
# first check run downloads the model files once; that takes a
# while and is normal.
import subprocess, sys
proc = subprocess.run([sys.executable, "/content/qwen_patching_run.py", "check"])
print()
print("check exit code:", proc.returncode)
if proc.returncode != 0:
    print("!" * 72)
    print("STOP. THE CHECK RUN FAILED. The runner's own output above says")
    print("where: a proof line, a loading candidate, the layer resolution,")
    print("the token check, or the feed proof.")
    print("Do not run Step 4. Show this output to Muse.")
    print("!" * 72)
    raise SystemExit("STOP: runner check failed")
print("check passed: proof lines right, model loaded, decoder layers")
print("resolved, token check passed with the audio positions derived")
print("and counted, feed proof passed on all 20 clips. Go on to Step 4.")
"""))

# ---------------------------------------------------------------- step 4
cells.append(md(r"""
## Step 4 · Run the surgery battery

The next cell runs the runner in **battery mode**: proof,
candidates, layer resolution, the token check, and the
feed proof again, then five stages.

**Baselines.** Every clip runs unpatched under its own
question: tone and silence under the beeping question,
speech under the speaking question. The yes score (the
probability of a yes token at the first generated
position, Series Two's scorer) and the answer word land
in `patching_baselines.csv`, with the open question's
answers on tone_00, silence_00, and speech_00 as context
rows.

**Captures.** Sixteen donor runs: the 8 tone clips and the
first 4 silence clips under the beeping question, the 4
speech clips under the speaking question. Each capture
saves the donor's full hidden states from one frozen
forward, with its row and its audio positions. Donor and
recipient in a direction always share the direction's
question, so a patch moves sound information, never
question information.

**The sweep.** Patched runs, one prefill and a first-token
read each, the hook set on one decoder layer at a time:

- tone into silence, 8 pairs, layers 0 to 35 (288 runs);
- speech into silence, 4 pairs, all layers (144 runs);
- silence into tone, 4 pairs, all layers (144 runs): the
  reverse control, where the score should fall;
- tone into the next tone clip, 4 pairs, all layers (144
  runs): the sham control, where the sound is already
  present and little should change;
- tone into silence at the non-audio positions only,
  4 pairs, layers 0, 7, 14, 21, 28, 35 (24 runs): the
  wrong-positions control, where no rescue is expected.

744 patched runs in all. Every run's row (pair, donor,
recipient, direction, layer, yes score, answer word) is
appended to `patching_results.csv` the moment it
finishes, so an interruption loses at most the run in
flight. A fresh battery first moves any earlier results
file aside with a timestamp, so runs never mix.

**The best-layer open-form check.** The runner averages
the tone patch's lift per layer (patched score minus each
recipient's own baseline, over the 8 pairs) and names the
best layer. At that layer, the 8 patched silence runs
answer the open question, and their answers land in
`patching_openform.csv`, with the unpatched silence_00
open answer printed beside them for comparison.

**The closing prints.** The runner ends by printing
"patching done", the pre-registered predictions from the
tracing plan quoted exactly, and the output filenames. It
prints no verdict on the predictions; the read happens
outside this notebook.
"""))

cells.append(code(r"""
# --- Battery mode: the surgery battery ---------------------------------------------------
# Same runner, same session Python, streamed live. The runner
# saves its outputs itself, to the Drive tracing folder and
# to /content. A nonzero exit code means the token check
# failed, the feed proof hard-stopped the run, or a patching
# mechanic failed (the banners above say which); the results
# CSV holds every patched run that finished before the stop.
import subprocess, sys
proc = subprocess.run([sys.executable, "/content/qwen_patching_run.py", "battery"])
print()
print("battery exit code:", proc.returncode)
if proc.returncode != 0:
    print("!" * 72)
    print("STOP. THE BATTERY RUN DID NOT FINISH CLEAN. Read the runner's")
    print("output above: the token check failed, the feed proof")
    print("hard-stopped the run, or a patching mechanic failed. The")
    print("results CSV holds every run that finished. Show it to Muse.")
    print("!" * 72)
    raise SystemExit("STOP: battery run not clean")
print("battery done: baselines taken, captures held, the sweep and")
print("its controls recorded, the best-layer open-form check saved.")
"""))

# ---------------------------------------------------------------- closing
cells.append(md(r"""
## Closing · What to send back

The files in the `tracing-qwen3b` Drive folder are the
record of this run:

- `patching_load_log.txt`: what loaded, and the full
  traceback of any candidate that failed.
- `patching_token_check.txt`: the histograms the audio
  positions were derived from, the settled id, and the
  counts. Read this before any result: it is the proof of
  which positions every capture and patch worked on.
- `patching_baselines.csv`: every clip's unpatched yes
  score and answer word, plus the three open answers.
- `patching_results.csv`: the 744 patched runs, appended
  as they finished.
- `patching_openform.csv`: the 8 patched silence runs'
  open answers at the best layer.

Send them back, and the surgery gets its read next to the
helpers and the attention watch, against the predictions
the plan registered before any of it was built. The
notebook itself prints no verdict.
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

out = os.path.join(HERE, "project_k_qwen_patching_colab.ipynb")
with open(out, "w") as fh:
    json.dump(nb, fh, indent=1)
print("wrote", out, "cells:", len(cells))
