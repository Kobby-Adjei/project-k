#!/usr/bin/env python3
"""Build project_k_qwen_tracing_colab.ipynb.

Tracing round 1 for Qwen2.5-Omni-3B: per-layer probes (the
"helper" classifiers) plus an attention readout. Activation
patching is NOT in this build; it is the follow-up build named
in the tracing plan (series-two-2026-11-02/tracing-plan.md),
and this notebook says so in its own text.

The design is the tracing plan's, implemented as written:

- The sanity gate comes before any extraction: one tone clip
  through the model's thinker with hidden states saved, the
  audio-token positions found from the model's own tokenizer
  and config (never a guessed id), counted, printed, and a
  hard stop if the count is zero.
- The probe clips are generated inside the runner with a
  fixed seed (20261103): 48 tone positives, 48 speech
  positives (the one probe sentence, shifted in time and
  gain, the documented augmentation), 48 shared negatives
  (12 silence, 18 noise bursts, 18 low rumbles). Every clip
  is 4.0 s at 16 kHz mono, measured and printed before use.
- Extraction saves, per clip and per layer, the mean-pooled
  hidden vector over the audio-token positions, plus the
  clip's label. The extraction count must equal the manifest
  count or the run hard-stops.
- Probes: one logistic regression per layer per task (tone
  vs negatives, speech vs negatives), trained on 32 positives
  plus 32 negatives, tested on the held-out 16 plus 16, with
  a shuffled-label control on the same split that must sit
  at chance.
- Attention readout: for a small subset of clips, the mean
  attention from the final sequence position to the
  audio-token positions, for the last 4 layers.

House pattern, mirrored from build_codec_qwen_notebook.py:
Step 1 is forked from the Series One Qwen notebook
(project_k_qwen_omni_colab.ipynb) in this same folder, with
the transformers 4.52.4 stack proof appended; the model
loading candidates are the codec runner's (3B fp16 first,
7B AWQ second), with full tracebacks written to a readable
load log; disable_talker; outputs are resume-safe; every
failure path prints a banner and stops with a distinct exit
code; no silent padding anywhere (a feed with missing audio
stops the run, exit 7 style).

One deliberate, printed deviation from the codec load call:
the candidates pass attn_implementation="eager". The
attention readout is a required output of this build, and
the default sdpa kernel does not return attention weights.
The sequences here are short, so the memory cost is small,
and the hidden-state math is the same. The runner prints
this reason in its proof lines.

Plain language throughout, no em dashes.
"""
import copy
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_NB = os.path.join(HERE, "project_k_qwen_omni_colab.ipynb")

# ------------------------------------------------------------ probe-set spec
# The counts and the probe CSV schema live here as literals
# so the build can be verified without running anything.
SEED = 20261103
N_TONE = 48
N_SPEECH = 48
N_NEGATIVES = 48          # 12 silence + 18 noise + 18 rumble
N_TRAIN_PER_CLASS = 32
N_TEST_PER_CLASS = 16
PROBE_CSV_FIELDS = ["layer", "task", "n_train", "n_test",
                    "accuracy", "shuffled_accuracy"]
ATTENTION_CSV_FIELDS = ["clip", "layer", "attention_to_audio"]
MANIFEST_FIELDS = ["clip_name", "group", "file", "freq_hz",
                   "start_s", "burst_s", "level_dbfs", "gain_db",
                   "prompt"]
assert N_TONE + N_SPEECH + N_NEGATIVES == 144
assert N_TRAIN_PER_CLASS + N_TEST_PER_CLASS == 48


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip().splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip().splitlines(True)}


with open(BASE_NB) as fh:
    base_nb = json.load(fh)

# Step 1 is the Series One Qwen notebook's own, reused the way
# the codec builder reused it: same session installs, same
# pins, plus the stack proof block (transformers exactly
# 4.52.4) and, new here, the scikit-learn proof the probes
# need.
step1_md = copy.deepcopy(base_nb["cells"][1])
step1_md["source"] = step1_md["source"] + (
    "\n"
    "**Tracing addition.** At the end, the cell proves the\n"
    "stack it just installed: it prints the imported\n"
    "transformers version and the file it loaded from, and\n"
    "stops the run if the version is not the pinned 4.52.4.\n"
    "It also proves scikit-learn, the library the helper\n"
    "classifiers are built with: if it is missing, the cell\n"
    "installs it, and either way it prints the version it\n"
    "found. This notebook's model runs under the session's\n"
    "own Python, so the versions printed here are the run's\n"
    "versions.\n"
).splitlines(True)

step1_code = copy.deepcopy(base_nb["cells"][2])
step1_code["source"] = step1_code["source"] + (
    "\n"
    "# --- Tracing addition: prove the installed stack --------------------------\n"
    "# The model and the probes run under this same session\n"
    "# Python, so the versions installed above are the versions\n"
    "# the run uses. Prove them here, and stop if the\n"
    "# transformers pin is not met.\n"
    "import transformers\n"
    "print(\"transformers version:\", transformers.__version__)\n"
    "print(\"transformers file:\", transformers.__file__)\n"
    "if transformers.__version__ != \"4.52.4\":\n"
    "    raise SystemExit(\"STOP: transformers is \"\n"
    "                     + transformers.__version__\n"
    "                     + \", not the pinned 4.52.4. Do not go on.\")\n"
    "print(\"stack proof passed: transformers 4.52.4, the Series One pin.\")\n"
    "\n"
    "# The helper classifiers are scikit-learn logistic\n"
    "# regressions. Colab sessions normally ship scikit-learn;\n"
    "# install it only if it is missing, and print the version\n"
    "# either way so the run record names what trained the\n"
    "# helpers.\n"
    "try:\n"
    "    import sklearn\n"
    "    print(\"scikit-learn version:\", sklearn.__version__)\n"
    "except ImportError:\n"
    "    print(\"scikit-learn missing; installing scikit-learn==1.6.1\")\n"
    "    subprocess.run([sys.executable, \"-m\", \"pip\", \"install\", \"-q\",\n"
    "                    \"scikit-learn==1.6.1\"], check=False)\n"
    "    import sklearn\n"
    "    print(\"scikit-learn version:\", sklearn.__version__)\n"
).splitlines(True)

cells = []
cells.append(md(r"""
# Project K, tracing round 1: Qwen2.5-Omni-3B

**Why this notebook exists.** Every Project K test so far
read the model's answers from outside. The answers say what
the model reported; they cannot say where, inside the model,
a sound stopped being carried. This notebook opens the model
and walks the sound through it. Qwen2.5-Omni-3B goes first
because its insides are standard: every layer's numbers (its
hidden states) can be saved while it listens.

**The goal.** Find the layer where the loud tone fades
inside the model, and test whether the speech is ever lost
at all. Round 1 carries two instruments: the helpers
(per-layer probes) and the attention watch. The third
instrument from the plan, surgery (activation patching), is
NOT in this notebook; it is the follow-up build, and it is
the causal check on whatever this round finds.

**The helpers, in plain words.** A helper is a tiny
classifier we train ourselves, on one layer's numbers only.
We play clips we chose, so we always know which clips carry
the sound and which do not. For each layer, the helper
learns the shape that layer's numbers take when the sound
is in them. Then it walks the layers on clips it never saw
in training. The layer where the helper stops finding the
sound is the neighborhood where the sound faded. The helper
only reads; it never changes the model, and the model is
never taught anything.

**What runs.** 144 short clips, generated inside the runner
with a fixed seed, each 4.0 seconds: 48 tone clips (a short
1000 Hz burst, pitch varied across 800, 1000, and 1200 Hz,
placed at different times and levels), 48 speech clips (the
one probe sentence, "The blue kettle boils at seven forty
two.", shifted in time and gain, the documented
augmentation for having a single sentence), and 48 negative
clips shared by both tasks (12 digital silence, 18 white
noise bursts, 18 low 120 Hz rumbles). Sound alone: no
picture rides along in this round, because the Series Two
2x2 showed the still picture is itself an interference
source for this model.

**The sanity gate comes first.** Before any clip set is
built, one tone clip goes through the model with its hidden
states saved. The runner finds the audio-token positions in
the model's input from the model's own tokenizer and config
(never a guessed number), prints the layer count, the
hidden size, the sequence length, and the audio-token
count, and stops dead if the audio-token count is zero.
Every later number depends on those positions being right,
so the gate is loud and early by design.

**Readout rules, fixed in advance** (from the tracing
plan):

- A helper failing at a layer means "not readable by this
  helper," which can also mean the information changed
  shape. No stage is declared lost on the helpers alone;
  the surgery build is the causal check.
- The shuffled-label control must sit at chance (about
  0.50). A control that beats chance means the helper is
  cheating on some side pattern, and the run is void until
  fixed.
- Predictions, registered before the run: the tone stays
  readable into the middle layers, then fades; the
  attention the answer pays to the tone's tokens is near
  zero; the speech stays readable at every layer, because
  its transcription is perfect, so its denial lives in how
  the answer is formed, past the information.

**Where it runs.** On the free Colab account (kadjei1),
under the routing rule: the 3B is a small model. **How to
run it.** Run the cells from the top, in order. Step 1 sets
up and proves the stack. Step 2 gets the three master
sounds. Step 3 writes the runner and runs its check mode
(proof, loading candidates, the sanity gate). Step 4 runs
battery mode: clip generation, extraction, the helpers, the
attention readout, and a plain-language tail.

**The one rule.** We claim only what we ran: these sounds,
this model, these layers, this run.
"""))
cells.append(step1_md)
cells.append(step1_code)

# ---------------------------------------------------------------- step 2
cells.append(md(r"""
## Step 2 · Get the three master sounds

Everything this run builds its clips from lives in your
Google Drive, in the folder named `codec-control`, the
codec control's folder. The next cell links your Drive,
finds that folder, and copies three files to a plain work
folder here:

- `codec_speech_raw.wav`: the probe sentence master, the
  swap study's own speech track, uncompressed.
- `codec_tone_ref_raw.wav`: the loud 1000 Hz tone master,
  the burst at the reference level, uncompressed.
- `codec_silence_raw.wav`: the silence master, true digital
  zeros, uncompressed.

The speech master is the source of all 48 speech clips; the
tone master is the sanity gate's clip; the silence master
is the record's proof of what the negative pool's silence
is. The runner re-measures everything from inside the
session before it believes any of it.

Do not go on until all three files say "ready". If any file
is missing, the cell stops and names it.
"""))

cells.append(code(r"""
# --- Get the three master sounds ---------------------------------------------------
import os, shutil, subprocess

FILES = ["codec_speech_raw.wav", "codec_tone_ref_raw.wav",
         "codec_silence_raw.wav"]
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

missing = []
for name in FILES:
    bring(name)
    path = os.path.join(WORK, name)
    if os.path.exists(path):
        d = duration_s(path)
        extra = f", {d:.3f} s" if d is not None else ""
        print("ready:  ", name, f"({os.path.getsize(path) / 1e6:.2f} MB{extra})")
    else:
        missing.append(name)
        print("MISSING:", name)
if missing:
    print()
    print("!" * 72)
    print("STOP. These master files are missing:")
    for name in missing:
        print("   ", name)
    print("Check the Drive folder name is exactly codec-control, or")
    print("upload the files by hand, then run this cell again.")
    print("Do not go on to Step 3.")
    print("!" * 72)
    raise SystemExit("STOP: master files missing in Step 2")
print()
print("ready: all 3 masters (speech, tone, silence)")
"""))

# ------------------------------------------------------------- runner script
# The runner is a plain Python script the notebook writes to
# /content/qwen_tracing_run.py and runs under the notebook
# session's own Python, the Qwen notebooks' own pattern.
RUNNER_SRC = r"""
# qwen_tracing_run.py - the tracing round 1 runner for
# Qwen2.5-Omni-3B. Written by project_k_qwen_tracing_colab.ipynb.
# It runs under the notebook session's own Python: the Qwen
# notebooks install their stack into the session (Step 1), and
# there is no private environment to run under. Modes: "check"
# runs the proof, the loading candidates, and the sanity gate,
# then stops. "battery" does all of that and then generates
# the probe clips, extracts the per-layer features, trains the
# helpers, reads the attention subset, and saves every output.
#
# Exit codes, one per failure family, all loud:
#   3  the stack proof failed (transformers is not 4.52.4)
#   4  both loading candidates failed
#   7  a feed returned no audio (never padded, never scored)
#   8  the sanity gate failed: no audio tokens found, or no
#      hidden states came back
#   9  probe clip generation failed its own measurements
#   10 the extraction count does not match the manifest count
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
import sklearn
print("scikit-learn version:", sklearn.__version__)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available()
      else "none")
print("attention implementation note: the loading candidates below pass")
print("attn_implementation='eager'. The attention readout is a required")
print("output of this build, and the default sdpa kernel does not return")
print("attention weights. The sequences here are short, so the memory")
print("cost is small, and the hidden-state math is the same.")

# --- Homes for the outputs -------------------------------------------------------
# The tracing outputs live in a Drive folder named
# tracing-qwen3b, created here if it is missing, with /content
# as the second home, the house pattern.
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

LOAD_LOG_PATHS = out_paths("tracing_qwen3b_load_log.txt")

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
log_load("Qwen2.5-Omni-3B tracing round 1 load log - "
         + datetime.datetime.now().isoformat(timespec="seconds")
         + " - runner mode: " + MODE + "\n")

# --- Load the model: the codec runner's candidates ---------------------------------
# Same candidates, same order, same talker-off call. The one
# printed difference is attn_implementation="eager" (see the
# proof lines above for why).
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
            repo, torch_dtype=torch.float16, device_map="auto",
            attn_implementation="eager")
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
# --- The feed path -----------------------------------------------------------------
# Sound alone, the codec control's Arm B form minus the
# picture: the conversation carries an audio element naming
# the WAV file and a text element, and no image and no video
# element. process_mm_info decodes the audio element with the
# same librosa call at 16 kHz mono it uses for a video
# soundtrack. The text is the Series Two presence question
# for the clip's sound, fixed per class: the tracing walks
# the model's insides while it hears the clip under the same
# question the outside tests asked. Negatives carry the tone
# question, the primary task's question, and the manifest
# records the exact prompt on every clip.
import json, math
import numpy as np
import librosa
import soundfile as sf
from qwen_omni_utils import process_mm_info

CLIPS = "/content/clips"
PROMPTS = {
    "tone": ("Is there the sound of a tone in this clip? "
             "Answer yes or no."),
    "speech": ("Is there the sound of speech in this clip? "
               "Answer yes or no."),
    "negative": ("Is there the sound of a tone in this clip? "
                 "Answer yes or no."),
}

class FeedError(RuntimeError):
    pass

class GateError(RuntimeError):
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
    # (the generated digital-silence negatives): an all-zeros
    # feed on a clip that should carry sound is a broken
    # feed, not a quiet clip, and scoring it anyway is how
    # silent failures get into the record.
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

def thinker_forward(inputs, want_attentions=False):
    # One frozen pass through the thinker with every layer's
    # numbers saved. The model is never trained, never asked
    # to generate; this is a reading, not a conversation.
    with torch.no_grad():
        out = THINKER(**inputs, output_hidden_states=True,
                      output_attentions=want_attentions)
    return out

# --- Find the audio tokens -----------------------------------------------------------
# The audio-token id is discovered, never guessed. Evidence,
# all printed: the tokenizer's tokens whose text mentions
# audio; the config attributes on the model config and the
# thinker config; and the count of each candidate inside a
# real input_ids row. The chosen id must decode to the audio
# pad token and must actually occur in the input. Anything
# less stops the run at the gate.
def find_audio_token_id(input_ids_row):
    tok = processor.tokenizer
    print("-" * 70)
    print("AUDIO TOKEN SEARCH")
    vocab = tok.get_vocab()
    audio_named = {t: i for t, i in vocab.items() if "audio" in t}
    for t in sorted(audio_named):
        print(f"  tokenizer token {t!r} -> id {audio_named[t]}")
    config_ids = {}
    for holder_name, holder in (
            ("model.config", model.config),
            ("thinker_config",
             getattr(model.config, "thinker_config", None))):
        if holder is None:
            continue
        for attr in ("audio_token_id", "audio_token_index"):
            val = getattr(holder, attr, None)
            if val is not None:
                config_ids[f"{holder_name}.{attr}"] = int(val)
                print(f"  {holder_name}.{attr} = {val}")
    candidates = {}
    pad_id = tok.convert_tokens_to_ids("<|audio_pad|>")
    if pad_id is not None and int(pad_id) >= 0:
        candidates[int(pad_id)] = "<|audio_pad|> (converted by name)"
    for t, i in audio_named.items():
        candidates[int(i)] = f"tokenizer token {t!r}"
    for where, val in config_ids.items():
        candidates.setdefault(val, f"config {where}")
    row = [int(x) for x in input_ids_row]
    scored = []
    for cid, why in sorted(candidates.items()):
        count = row.count(cid)
        decoded = tok.decode([cid])
        print(f"  candidate id {cid} ({why}): decodes to "
              f"{decoded!r}, occurs {count} time(s) in the "
              f"sanity input")
        scored.append((cid, why, decoded, count))
    # The pad token first: it is the placeholder the
    # processor expands into the sequence where the audio
    # sits, so its positions ARE the audio-token positions.
    for cid, why, decoded, count in scored:
        if decoded == "<|audio_pad|>" and count > 0:
            print(f"AUDIO TOKEN ID: {cid} ({why}), decodes to "
                  f"{decoded!r}, {count} position(s) in the input")
            return cid, count
    for cid, why, decoded, count in scored:
        if "audio" in decoded and count > 0:
            print(f"AUDIO TOKEN ID: {cid} ({why}), decodes to "
                  f"{decoded!r}, {count} position(s) in the input")
            return cid, count
    raise GateError(
        "no candidate audio token occurs in the sanity input; "
        f"candidates were {[(c, d, n) for c, _, d, n in scored]}")

AUDIO_TOKEN_ID = None   # set by the sanity gate

def audio_positions(input_ids_row):
    row = [int(x) for x in input_ids_row]
    return [i for i, x in enumerate(row) if x == AUDIO_TOKEN_ID]

# --- The sanity gate -------------------------------------------------------------------
# One tone clip through the thinker, before anything else is
# built. Prints the shape of what came back and stops dead
# (exit 8) if the audio tokens are not found or the hidden
# states are missing. Loud and early, by design: every later
# number in this run depends on these positions being right.
def sanity_gate():
    global AUDIO_TOKEN_ID
    tone_path = os.path.join(CLIPS, "codec_tone_ref_raw.wav")
    print("=" * 70)
    print("SANITY GATE: one tone clip through the thinker")
    print("gate clip:", tone_path)
    try:
        inputs, audio_np = feed_inputs(tone_path, PROMPTS["tone"])
    except FeedError as e:
        print()
        print("!" * 70)
        print("SANITY GATE FEED FAILURE:")
        print(str(e))
        print("The gate clip's feed returned no audio. No extraction")
        print("has run. Show this whole output to Muse.")
        print("!" * 70)
        sys.exit(7)
    print(f"gate feed: decoded waveform {len(audio_np)} samples "
          f"at 16 kHz ({len(audio_np) / 16000:.2f} s)")
    row = inputs.input_ids[0].tolist()
    try:
        AUDIO_TOKEN_ID, gate_count = find_audio_token_id(row)
    except GateError as e:
        print()
        print("!" * 70)
        print("SANITY GATE FAILED:")
        print(str(e))
        print("Without the audio-token positions, every later number")
        print("in this run would be built on sand. No extraction has")
        print("run. Show this whole output to Muse.")
        print("!" * 70)
        sys.exit(8)
    out = thinker_forward(inputs)
    hs = getattr(out, "hidden_states", None)
    if hs is None or len(hs) == 0:
        print()
        print("!" * 70)
        print("SANITY GATE FAILED: the thinker returned no hidden")
        print("states. No extraction has run. Show this whole output")
        print("to Muse.")
        print("!" * 70)
        sys.exit(8)
    n_states = len(hs)
    seq_len = int(hs[0].shape[1])
    hidden = int(hs[0].shape[2])
    print(f"hidden states returned: {n_states} tensors "
          f"(layer 0 is the embedding output, then one per "
          f"thinker layer)")
    print(f"thinker layers: {n_states - 1}")
    print(f"hidden size: {hidden}")
    print(f"sequence length: {seq_len}")
    print(f"audio-token count: {gate_count}")
    print(f"audio-token positions: first "
          f"{audio_positions(row)[:5]}, last "
          f"{audio_positions(row)[-5:]}")
    if gate_count == 0:
        print()
        print("!" * 70)
        print("SANITY GATE FAILED: audio-token count is zero.")
        print("!" * 70)
        sys.exit(8)
    print("SANITY GATE PASSED: the feed carries audio, the thinker")
    print("returns its layers, and the audio tokens are located and")
    print("counted. The positions every later number uses are these.")
    del out, inputs
    free_memory()
    return n_states, hidden

N_STATES, HIDDEN_SIZE = sanity_gate()

if MODE == "check":
    print()
    print("check mode done: proof, candidates, and the sanity gate")
    print("all passed. No clips were generated and no extraction ran.")
    print("Run Step 4 for the battery.")
    sys.exit(0)
"""

RUNNER_SRC += r"""
# --- Probe clip generation -------------------------------------------------------------
# Fixed seed, so the run can be repeated exactly and the
# split can be re-derived. 144 clips, each exactly 4.0 s at
# 16 kHz mono (64000 samples). Tone positives vary pitch,
# placement, and level. Speech positives are the one probe
# sentence (the master's active region), shifted in time and
# gain: the documented augmentation for having a single
# sentence. Negatives are a shared pool: digital silence,
# white-noise bursts, low sine rumble. Every clip is measured
# (sample count, RMS) and printed before anything uses it,
# and a failed measurement stops the run (exit 9).
import csv, io

SEED = __SEED__
SR = 16000
CLIP_SAMPLES = SR * 4
CLIP_DIRS = ["/content/drive/MyDrive/tracing-qwen3b/tracing_clips",
             "/content/tracing_clips"]
for d in CLIP_DIRS:
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
MANIFEST_FIELDS = __MANIFEST_FIELDS__

def rms_db_np(x):
    if len(x) == 0:
        return -120.0
    v = float(np.sqrt(np.mean(np.square(x.astype(np.float64)))))
    return 20.0 * math.log10(v) if v > 0 else -120.0

def db_to_amp(db):
    return float(10.0 ** (db / 20.0))

def write_clip(name, wave):
    wave = np.asarray(wave, dtype=np.float32)
    assert len(wave) == CLIP_SAMPLES, \
        f"clip {name} has {len(wave)} samples, not {CLIP_SAMPLES}"
    saved = []
    for d in CLIP_DIRS:
        p = os.path.join(d, name)
        try:
            if not os.path.exists(p):
                sf.write(p, wave, SR, subtype="PCM_16")
            saved.append(p)
        except Exception:
            pass
    # The extraction reads from /content first: a plain local
    # path the feed cannot miss.
    local = os.path.join("/content/tracing_clips", name)
    return local

def speech_segment():
    # The sentence inside the speech master: the region whose
    # short-window RMS rises above -45 dBFS, with a small pad.
    path = os.path.join(CLIPS, "codec_speech_raw.wav")
    wave, _ = librosa.load(path, sr=SR, mono=True)
    wave = np.asarray(wave, dtype=np.float32)
    win = SR // 100  # 10 ms
    n = len(wave) // win
    rms = np.array([rms_db_np(wave[i * win:(i + 1) * win])
                    for i in range(n)])
    above = np.where(rms > -45.0)[0]
    if len(above) == 0:
        raise RuntimeError(
            "no active region found in the speech master; the "
            "speech positives cannot be built")
    start = max(0, int(above[0] * win) - SR // 20)
    end = min(len(wave), int((above[-1] + 1) * win) + SR // 20)
    seg = wave[start:end]
    print(f"speech segment: {len(seg) / SR:.2f} s cut from the "
          f"master at {start / SR:.2f} s to {end / SR:.2f} s")
    if len(seg) > CLIP_SAMPLES:
        seg = seg[:CLIP_SAMPLES]
        print("speech segment trimmed to the 4.0 s window")
    return seg

def generate_clips():
    rng = np.random.default_rng(SEED)
    rows = []
    # ---- tone positives (48)
    for i in range(48):
        freq = (800, 1000, 1200)[i % 3]
        start_s = float(rng.uniform(0.5, 3.0))
        level = float(rng.uniform(-26.0, -14.0))
        burst_s = 0.5
        wave = np.zeros(CLIP_SAMPLES, dtype=np.float32)
        n_burst = int(burst_s * SR)
        t = np.arange(n_burst) / SR
        burst = (db_to_amp(level)
                 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        edge = SR // 200  # 5 ms raised-cosine edges, no clicks
        ramp = 0.5 - 0.5 * np.cos(
            np.linspace(0, np.pi, edge)).astype(np.float32)
        burst[:edge] *= ramp
        burst[-edge:] *= ramp[::-1]
        s0 = int(start_s * SR)
        wave[s0:s0 + n_burst] += burst
        name = f"tone_{i:02d}.wav"
        path = write_clip(name, wave)
        rows.append({
            "clip_name": name[:-4], "group": "tone", "file": path,
            "freq_hz": freq, "start_s": round(start_s, 3),
            "burst_s": burst_s, "level_dbfs": round(level, 2),
            "gain_db": "", "prompt": PROMPTS["tone"],
            "_wave": wave})
    # ---- speech positives (48)
    seg = speech_segment()
    max_offset = max(0.0, min(1.5, 4.0 - len(seg) / SR))
    for i in range(48):
        offset = float(rng.uniform(0.0, max_offset)) \
            if max_offset > 0 else 0.0
        gain = float(rng.uniform(-6.0, 3.0))
        wave = np.zeros(CLIP_SAMPLES, dtype=np.float32)
        placed = (seg * db_to_amp(gain)).astype(np.float32)
        peak = float(np.max(np.abs(placed))) if len(placed) else 0.0
        if peak > 0.95:
            placed = (placed * (0.95 / peak)).astype(np.float32)
            print(f"speech_{i:02d}: gain {gain:.2f} dB would peak at "
                  f"{peak:.2f}; scaled to a 0.95 peak instead")
        s0 = int(offset * SR)
        wave[s0:s0 + len(placed)] += placed
        name = f"speech_{i:02d}.wav"
        path = write_clip(name, wave)
        rows.append({
            "clip_name": name[:-4], "group": "speech", "file": path,
            "freq_hz": "", "start_s": round(offset, 3),
            "burst_s": round(len(seg) / SR, 3), "level_dbfs": "",
            "gain_db": round(gain, 2), "prompt": PROMPTS["speech"],
            "_wave": wave})
    # ---- negatives (48, shared pool)
    for i in range(12):
        wave = np.zeros(CLIP_SAMPLES, dtype=np.float32)
        name = f"neg_silence_{i:02d}.wav"
        path = write_clip(name, wave)
        rows.append({
            "clip_name": name[:-4], "group": "negative", "file": path,
            "freq_hz": "", "start_s": "", "burst_s": "",
            "level_dbfs": "", "gain_db": "",
            "prompt": PROMPTS["negative"], "_wave": wave})
    for i in range(18):
        level = float(rng.uniform(-30.0, -18.0))
        burst_s = float(rng.uniform(0.5, 2.0))
        start_s = float(rng.uniform(0.0, 4.0 - burst_s))
        wave = np.zeros(CLIP_SAMPLES, dtype=np.float32)
        n_burst = int(burst_s * SR)
        noise = rng.standard_normal(n_burst).astype(np.float32)
        cur = float(np.sqrt(np.mean(np.square(
            noise.astype(np.float64)))))
        noise = (noise * (db_to_amp(level) / cur)).astype(np.float32)
        s0 = int(start_s * SR)
        wave[s0:s0 + n_burst] += noise
        name = f"neg_noise_{i:02d}.wav"
        path = write_clip(name, wave)
        rows.append({
            "clip_name": name[:-4], "group": "negative", "file": path,
            "freq_hz": "", "start_s": round(start_s, 3),
            "burst_s": round(burst_s, 3), "level_dbfs": round(level, 2),
            "gain_db": "", "prompt": PROMPTS["negative"],
            "_wave": wave})
    for i in range(18):
        level = float(rng.uniform(-24.0, -16.0))
        wave = np.zeros(CLIP_SAMPLES, dtype=np.float32)
        t = np.arange(CLIP_SAMPLES) / SR
        wave += (db_to_amp(level)
                 * np.sin(2 * np.pi * 120 * t)).astype(np.float32)
        name = f"neg_rumble_{i:02d}.wav"
        path = write_clip(name, wave)
        rows.append({
            "clip_name": name[:-4], "group": "negative", "file": path,
            "freq_hz": 120, "start_s": 0.0, "burst_s": 4.0,
            "level_dbfs": round(level, 2), "gain_db": "",
            "prompt": PROMPTS["negative"], "_wave": wave})
    # ---- measure every clip before anything uses it
    print("-" * 70)
    print(f"generated {len(rows)} clips; measuring each one:")
    for r in rows:
        w = r.pop("_wave")
        n = len(w)
        level = rms_db_np(w)
        print(f"  {r['clip_name']:<16} group={r['group']:<8} "
              f"samples={n} duration={n / SR:.3f} s "
              f"rms={level:.2f} dB")
        if n != CLIP_SAMPLES:
            print()
            print("!" * 70)
            print(f"CLIP MEASUREMENT FAILED: {r['clip_name']} has "
                  f"{n} samples, not {CLIP_SAMPLES}. The generated")
            print("set is not what the plan says. Nothing was")
            print("extracted. Show this whole output to Muse.")
            print("!" * 70)
            sys.exit(9)
    def write_manifest(p):
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in MANIFEST_FIELDS})
    save_both("tracing_manifest.csv", write_manifest)
    return rows

# --- Extraction --------------------------------------------------------------------------
# One frozen forward pass per clip. Per layer, the vectors at
# the audio-token positions are averaged into one vector per
# clip (the layer's sketch of the sound, in the plan's plain
# words) and stored float32. The attention subset (the first
# 6 tone and first 6 negative clips, in manifest order) gets
# a second forward with attentions returned; for its last 4
# layers we record the mean attention from the final sequence
# position to the audio-token positions.
FEATURES_NAME = "tracing_features.npz"

def find_features():
    for p in out_paths(FEATURES_NAME):
        if os.path.exists(p):
            return p
    return None

def extract(rows):
    n_clips = len(rows)
    feats = np.zeros((N_STATES, n_clips, HIDDEN_SIZE),
                     dtype=np.float32)
    audio_counts = np.zeros(n_clips, dtype=np.int64)
    seq_lens = np.zeros(n_clips, dtype=np.int64)
    attention_rows = []
    subset = set()
    tone_seen = neg_seen = 0
    for idx, r in enumerate(rows):
        if r["group"] == "tone" and tone_seen < 6:
            subset.add(idx)
            tone_seen += 1
        elif r["group"] == "negative" and neg_seen < 6:
            subset.add(idx)
            neg_seen += 1
    for idx, r in enumerate(rows):
        expect_silence = r["clip_name"].startswith("neg_silence")
        try:
            inputs, audio_np = feed_inputs(
                r["file"], r["prompt"],
                expect_silence=expect_silence)
        except FeedError as e:
            print()
            print("!" * 70)
            print("EXTRACTION FEED HARD STOP:")
            print(str(e))
            print(f"Clip {idx + 1} of {n_clips} ({r['clip_name']}) "
                  f"was not extracted, and the run stops here. No")
            print("padding was substituted. Show this whole output")
            print("to Muse.")
            print("!" * 70)
            sys.exit(7)
        row_ids = inputs.input_ids[0].tolist()
        pos = audio_positions(row_ids)
        if len(pos) == 0:
            print()
            print("!" * 70)
            print(f"EXTRACTION HARD STOP: clip {r['clip_name']} has "
                  f"zero audio-token positions under the gate's")
            print("audio token id. The gate found them on the tone")
            print("master, so this clip's feed differs somehow. No")
            print("vector was faked. Show this output to Muse.")
            print("!" * 70)
            sys.exit(8)
        out = thinker_forward(inputs)
        hs = out.hidden_states
        pos_t = torch.tensor(pos, dtype=torch.long)
        for li in range(N_STATES):
            layer_vec = hs[li][0].index_select(0, pos_t.to(
                hs[li].device)).mean(dim=0)
            feats[li, idx] = layer_vec.float().cpu().numpy()
        audio_counts[idx] = len(pos)
        seq_lens[idx] = int(hs[0].shape[1])
        print(f"extracted {idx + 1}/{n_clips}: {r['clip_name']} "
              f"(group={r['group']}, audio tokens={len(pos)}, "
              f"seq len={seq_lens[idx]})")
        del out, inputs
        if idx in subset:
            inputs2, _ = feed_inputs(
                r["file"], r["prompt"],
                expect_silence=expect_silence)
            out2 = thinker_forward(inputs2, want_attentions=True)
            atts = getattr(out2, "attentions", None)
            if atts is None or len(atts) == 0:
                print(f"WARNING: attention subset clip "
                      f"{r['clip_name']} returned no attentions; "
                      f"its rows will be missing from the attention")
                print("CSV, and this warning is the record of why.")
            else:
                n_attn_layers = len(atts)
                for li in range(n_attn_layers - 4, n_attn_layers):
                    a = atts[li][0].float()  # heads, seq, seq
                    val = float(a[:, -1, :][:, pos].mean().item())
                    attention_rows.append({
                        "clip": r["clip_name"], "layer": li,
                        "attention_to_audio": round(val, 6)})
                print(f"attention read: {r['clip_name']} "
                      f"(last 4 of {n_attn_layers} layers)")
            del out2, inputs2
        free_memory()
    extracted = int(np.count_nonzero(audio_counts))
    print(f"extraction count: {extracted} of {n_clips} clips")
    if extracted != n_clips:
        print()
        print("!" * 70)
        print(f"EXTRACTION COUNT MISMATCH: {extracted} clips have "
              f"features, the manifest lists {n_clips}. The run")
        print("stops; partial features are not saved as a result.")
        print("!" * 70)
        sys.exit(10)
    groups = np.array([r["group"] for r in rows])
    names = np.array([r["clip_name"] for r in rows])

    def write_npz(p):
        np.savez(p, features=feats, groups=groups, clip_names=names,
                 audio_token_counts=audio_counts, seq_lens=seq_lens)
    save_both(FEATURES_NAME, write_npz)
    return feats, groups, names, attention_rows

# --- Battery flow ---------------------------------------------------------------------------
rows = generate_clips()
attention_rows = []
feats = groups = names = None
existing = find_features()
if existing is not None:
    try:
        z = np.load(existing, allow_pickle=False)
        f0 = z["features"]
        g0 = [str(x) for x in z["groups"]]
        n0 = [str(x) for x in z["clip_names"]]
        if f0.shape == (N_STATES, len(rows), HIDDEN_SIZE) \
                and n0 == [r["clip_name"] for r in rows] \
                and g0 == [r["group"] for r in rows]:
            feats, groups, names = f0, z["groups"], z["clip_names"]
            print(f"resume: {existing} already holds all "
                  f"{len(rows)} clips with matching names and "
                  f"groups; extraction skipped, going straight "
                  f"to the probes. The attention readout is not "
                  f"re-run on a resume; its CSV from the first "
                  f"pass stands.")
        else:
            print(f"resume: {existing} exists but does not match "
                  f"this manifest (shape {f0.shape}); extracting "
                  f"fresh and overwriting it.")
    except Exception as e:
        print(f"resume: could not read {existing} ({str(e)[:120]}); "
              f"extracting fresh.")
if feats is None:
    feats, groups, names, attention_rows = extract(rows)

if attention_rows:
    def write_attention(p):
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=__ATTENTION_FIELDS__)
            w.writeheader()
            w.writerows(attention_rows)
    save_both("tracing_attention.csv", write_attention)

# --- The helpers (probes) ---------------------------------------------------------------------
# One logistic regression per layer per task. The split is by
# clip index within each class, shuffled with the run seed:
# the first 32 of each class train, the last 16 test, the
# same split at every layer and for both tasks, printed here
# so it can be re-derived. The control trains on the same
# training vectors with the labels shuffled (seed plus the
# layer), and is tested on the true test labels: a control
# that beats chance means the helper is reading some side
# pattern, and the plan's rule voids the run.
from sklearn.linear_model import LogisticRegression

PROBE_FIELDS = __PROBE_FIELDS__
group_list = [str(g) for g in groups]
class_indices = {}
for gname in ("tone", "speech", "negative"):
    idxs = [i for i, g in enumerate(group_list) if g == gname]
    rng_split = np.random.default_rng(SEED)
    order = rng_split.permutation(idxs)
    class_indices[gname] = ([int(x) for x in order[:32]],
                            [int(x) for x in order[32:]])
    print(f"split for {gname}: {len(class_indices[gname][0])} "
          f"train, {len(class_indices[gname][1])} test "
          f"(seed {SEED})")

probe_rows = []
for task, pos_group in (("tone_vs_negatives", "tone"),
                        ("speech_vs_negatives", "speech")):
    tr_pos, te_pos = class_indices[pos_group]
    tr_neg, te_neg = class_indices["negative"]
    tr_idx = tr_pos + tr_neg
    te_idx = te_pos + te_neg
    y_train = np.array([1] * len(tr_pos) + [0] * len(tr_neg))
    y_test = np.array([1] * len(te_pos) + [0] * len(te_neg))
    for li in range(N_STATES):
        X = feats[li]
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X[tr_idx], y_train)
        acc = float(clf.score(X[te_idx], y_test))
        rng_shuf = np.random.default_rng(SEED + 1000 + li)
        y_shuf = rng_shuf.permutation(y_train)
        clf2 = LogisticRegression(max_iter=1000, C=1.0)
        clf2.fit(X[tr_idx], y_shuf)
        acc_shuf = float(clf2.score(X[te_idx], y_test))
        probe_rows.append({
            "layer": li, "task": task,
            "n_train": len(tr_idx), "n_test": len(te_idx),
            "accuracy": round(acc, 4),
            "shuffled_accuracy": round(acc_shuf, 4)})
        print(f"probe {task} layer {li}: accuracy {acc:.3f}, "
              f"shuffled control {acc_shuf:.3f}")

def write_probes(p):
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=PROBE_FIELDS)
        w.writeheader()
        w.writerows(probe_rows)
save_both("tracing_probe_results.csv", write_probes)

# --- The plain-language tail ---------------------------------------------------------------------
print()
print("=" * 70)
print("READOUT (the plan's rules, applied mechanically)")
for task in ("tone_vs_negatives", "speech_vs_negatives"):
    accs = [r["accuracy"] for r in probe_rows if r["task"] == task]
    fade = "never"
    for li in range(len(accs)):
        if accs[li] < 0.70 and all(a < 0.70 for a in accs[li:]):
            fade = str(li)
            break
    print(f"  {task}: first layer where the helper falls below "
          f"0.70 and stays below: {fade}")
shufs = [r["shuffled_accuracy"] for r in probe_rows]
print(f"  shuffled controls across all layers and tasks: "
      f"min {min(shufs):.3f}, max {max(shufs):.3f} "
      f"(chance is about 0.50; a control well above chance "
      f"voids the run under the plan's rules)")
print("  layer 0 is the embedding output; layer k is the output")
print("  of thinker layer k. A fade is where this helper stops")
print("  reading the sound, not proof the sound is gone: the")
print("  surgery build is the causal check, per the plan.")
print("=" * 70)
print("battery done: clips generated and measured, features")
print("extracted and counted, helpers trained with controls,")
print("attention read on the subset. Outputs: "
      "tracing_manifest.csv, tracing_features.npz, "
      "tracing_probe_results.csv, tracing_attention.csv.")
"""

RUNNER_SRC = RUNNER_SRC.replace("__SEED__", str(SEED))
RUNNER_SRC = RUNNER_SRC.replace("__MANIFEST_FIELDS__", repr(MANIFEST_FIELDS))
RUNNER_SRC = RUNNER_SRC.replace("__PROBE_FIELDS__", repr(PROBE_CSV_FIELDS))
RUNNER_SRC = RUNNER_SRC.replace("__ATTENTION_FIELDS__",
                                repr(ATTENTION_CSV_FIELDS))
for _ph in ("__SEED__", "__MANIFEST_FIELDS__", "__PROBE_FIELDS__",
            "__ATTENTION_FIELDS__"):
    assert _ph not in RUNNER_SRC, f"placeholder survived: {_ph}"

# ---------------------------------------------------------------- step 3
cells.append(md(r"""
## Step 3 · Write the runner, then check: proof, candidates, sanity gate

The model never runs inside this notebook's cells directly.
It runs in a plain script, `/content/qwen_tracing_run.py`,
under this session's own Python, the Qwen notebooks'
pattern. The first cell below writes that script. The
second runs it in **check mode**.

Check mode does three things, in order:

1. **Proof again.** The versions and files the runner
   imported, with the transformers pin checked a second
   time: exactly 4.52.4, or the run stops. The scikit-learn
   version prints here too, and one note explains the single
   deliberate difference from the earlier runners' load
   call: the candidates load with eager attention, because
   this build must return attention weights and the default
   sdpa kernel does not.
2. **The loading candidates.** The codec runner's list: the
   3B in 16-bit form (float16) first, the 7B AWQ build
   second, only if the small one fails. The talker is turned
   off on whichever loads. A failed candidate prints its
   full traceback and appends the same words to
   `tracing_qwen3b_load_log.txt` in the tracing folder.
3. **The sanity gate.** One tone clip (the tone master from
   Step 2) goes through the model's thinker with hidden
   states saved. The runner searches the model's own
   tokenizer and config for the audio token, prints every
   candidate with its decoded form and its count inside the
   gate input, and takes its positions from the audio pad
   token that actually occurs there. It prints the layer
   count, the hidden size, the sequence length, and the
   audio-token count, and stops dead (exit code 8) if the
   count is zero or the hidden states are missing. No clips
   are generated and no extraction runs in check mode.
"""))

cells.append(code(
    "# --- Write the tracing runner script --------------------------------------------\n"
    "# The runner holds the proof, the loading candidates, the\n"
    "# sanity gate, the clip generation, the extraction, the\n"
    "# helpers, and the attention readout. It is written to a\n"
    "# plain file here and only ever run under this session's\n"
    "# own Python, the interpreter Step 1 installed into.\n"
    "RUNNER_SOURCE = " + repr(RUNNER_SRC) + "\n"
    "with open(\"/content/qwen_tracing_run.py\", \"w\") as fh:\n"
    "    fh.write(RUNNER_SOURCE)\n"
    "print(\"runner written:\", \"/content/qwen_tracing_run.py\", "
    "len(RUNNER_SOURCE), \"chars\")\n"))

cells.append(code(r"""
# --- Check mode: proof, candidates, sanity gate --------------------------------------
# Runs the runner under this session's own Python and streams
# its output here live. A nonzero exit code means the proof, a
# loading candidate, or the sanity gate failed, and the run
# stops in this cell. The first check run downloads the model
# files once; that takes a while and is normal.
import subprocess, sys
proc = subprocess.run([sys.executable, "/content/qwen_tracing_run.py", "check"])
print()
print("check exit code:", proc.returncode)
if proc.returncode != 0:
    print("!" * 72)
    print("STOP. THE CHECK RUN FAILED. The runner's own output above says")
    print("where: a proof line, a loading candidate, or the sanity gate.")
    print("Do not run Step 4. Show this output to Muse.")
    print("!" * 72)
    raise SystemExit("STOP: runner check failed")
print("check passed: proof lines right, model loaded, sanity gate")
print("passed with the audio tokens located and counted. Go on to Step 4.")
"""))

# ---------------------------------------------------------------- step 4
cells.append(md(r"""
## Step 4 · Run the tracing battery

The next cell runs the runner in **battery mode**: proof,
candidates, and the sanity gate again, then four stages.

**Clip generation.** 144 clips are synthesized with the
fixed seed 20261103, each exactly 4.0 seconds at 16 kHz:
48 tone, 48 speech, 48 negatives (12 silence, 18 noise
bursts, 18 rumbles), with the parameters of every clip
recorded in `tracing_manifest.csv`. Every clip is measured
and printed (sample count, duration, RMS) before anything
uses it; a clip that fails its measurement stops the run.

**Extraction.** Each clip goes through the frozen model
once, sound alone, under the Series Two presence question
for its sound. Any feed that returns no audio, or decodes
to zeros on a clip that should carry sound, stops the run
on the spot (exit code 7): nothing is padded and nothing is
scored on a broken feed. Per layer, the vectors at the
audio-token positions are averaged into one vector per
clip and saved to `tracing_features.npz`. The extraction
count must equal the manifest count, 144, or the run stops
(exit code 10). If the features file already holds all 144
clips with matching names, extraction is skipped and the
run goes straight to the helpers.

**The helpers.** Per layer and per task (tone vs
negatives, speech vs negatives), a logistic regression
trains on 32 positives plus 32 negatives and is tested on
the held-out 16 plus 16, the same split at every layer.
Each helper is shadowed by a shuffled-label control on the
same split. Results land in `tracing_probe_results.csv`:
one row per layer per task, controls included.

**The attention readout.** The first 6 tone and first 6
negative clips get a second pass with attentions returned.
For the last 4 layers, the mean attention from the final
sequence position to the audio-token positions lands in
`tracing_attention.csv`.

**The tail.** The runner ends with a plain-language
readout: for each task, the first layer where the helper
falls below 0.70 and stays below (or "never"), and the
shuffled controls' range across the whole run.
"""))

cells.append(code(r"""
# --- Battery mode: the tracing round 1 battery -----------------------------------------
# Same runner, same session Python, streamed live. The runner
# saves its outputs itself, to the Drive tracing folder and
# to /content. A nonzero exit code means a gate failed, a
# feed hard-stopped the run, clip generation failed its
# measurements, or the extraction count mismatched (the
# banners above say which).
import subprocess, sys
proc = subprocess.run([sys.executable, "/content/qwen_tracing_run.py", "battery"])
print()
print("battery exit code:", proc.returncode)
if proc.returncode != 0:
    print("!" * 72)
    print("STOP. THE BATTERY RUN DID NOT FINISH CLEAN. Read the runner's")
    print("output above: a gate failed, a feed proof hard-stopped the run,")
    print("clip generation failed, or the counts mismatched. Show it to")
    print("Muse.")
    print("!" * 72)
    raise SystemExit("STOP: battery run not clean")
print("battery done: features extracted and counted, helpers trained")
print("with controls, attention read, CSVs saved.")
"""))

# ---------------------------------------------------------------- closing
cells.append(md(r"""
## Closing · How to read the answers

Read each task's curve on its own first: one accuracy per
layer, from the embedding output (layer 0) to the last
thinker layer, with the shuffled control beside it.

- **The tone fades somewhere.** The fade layer is the
  neighborhood where this helper stops reading the tone.
  Read it with the attention numbers beside it: if the
  final position's attention to the tone's tokens was near
  zero all along, the answer never looked; if attention was
  healthy and the report still missed, it looked and
  dropped. Either way, the fade is a readability result,
  not yet a proof of loss. The surgery build is the causal
  check that confirms or kills the claim.
- **The speech reads at every layer.** Then nothing about
  the speech is lost inside this model, and its denial in
  the outside tests lives in how the presence answer is
  formed, past the information. That was the registered
  prediction.
- **The speech fades somewhere.** Then the prediction dies
  in public, the taxonomy changes, and the write-up says so
  plainly. A result that kills a prediction is a result.
- **A control beats chance.** The run is void until fixed,
  under the plan's rules, whatever the curves look like.
  A helper that can read shuffled labels is reading some
  side pattern in the clips, not the sound.
- **Scope.** Every claim stays capped at this model, these
  sounds, this question per class, this run. Pegasus stays
  a black box; MiniCPM and OLA tracing wait until these
  instruments have run end to end here; the picture
  condition is round 2.

When you finish, the files in the `tracing-qwen3b` Drive
folder are the record: the manifest, the features, the
probe results, and the attention readout. Send them back,
and the tracing gets written up next to the codec control.
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

out = os.path.join(HERE, "project_k_qwen_tracing_colab.ipynb")
with open(out, "w") as fh:
    json.dump(nb, fh, indent=1)
print("wrote", out, "cells:", len(cells))
