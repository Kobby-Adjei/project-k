#!/usr/bin/env python3
"""Build project_k_qwen_property_colab.ipynb.

Tracing round 2 for Qwen2.5-Omni-3B: the round 1 instruments
(per-layer probes, the "helper" classifiers, plus the
attention readout) with the probe design fixed, after round
1's readout was voided by its own shuffled-label control
(see tracing-qwen3b/tracing-round1-read.md). The method and
the pre-registered predictions are unchanged from
series-two-2026-11-02/tracing-plan.md; the rule stands: a
shuffled control that beats chance voids the readout.

The round 2 changes, and no others in spirit (the round 1
read's fix list, followed exactly):

1. TOKEN CHECK FIRST. Before any extraction, one tone clip
   and one speech clip (the codec masters) go through the
   processor, and the step prints the full histogram of
   token ids in each processed input row (every id with
   count > 0, decoded names for special tokens), the
   processor's audio feature lengths, the candidate audio
   ids with their counts, and the positions the notebook
   will pool over. The pooling id is DERIVED from the
   histogram: the candidate that occurs at audio-frame
   scale (the highest count), never the first candidate
   that merely occurs, which is what round 1's gate did.
   If the derived audio-position count is under 10 for a
   clip, the step prints TOKEN CHECK FAILED and stops the
   run loudly (exit 11) before any extraction. The full
   transcript is saved as property_token_check.txt.

2. BIGGER BATTERY. 360 clips per class (tone, speech,
   silence), 420 total, 4.0 s each, generated synthetically
   in the notebook with a fixed seed, raw WAV fed directly
   (no MP4, no codec anywhere; the codec control's raw arm
   form). Variation inside each class so a helper must read
   the sound, not memorize one waveform: tone varies start
   (0.5 to 3.0 s), level (-24 to -14 dBFS), frequency jitter
   (950 to 1050 Hz); speech is the probe sentence rendered
   from the master with level variation, tape-style speed
   variation (0.92 to 1.08, which shifts pitch with it),
   and start offsets; silence is true digital silence plus
   near-silence noise-floor variants. Every class draws its
   clip backgrounds from the same scheme (digital silence,
   with a one-in-six chance of a -75 to -65 dBFS noise
   floor), so the background cannot be a class cue. Every
   clip's per-second RMS and waveform hash are printed in
   the extraction log (the feed proof). Fixed split by
   seed: 240 train / 120 held-out test per class, the same
   split for every layer and both tasks.

3. SMALLER, REGULARIZED READOUT. Per layer, mean-pool the
   hidden states over the verified audio positions only.
   Two readouts, both reported: logistic regression with
   strong regularization (C=0.01) on standardized features,
   and the same probe after PCA to 64 components. For every
   layer, task, and readout, the results CSV carries the
   real test accuracy and the shuffled-label control
   accuracy as the mean of 5 shuffles trained and tested
   identically (with the shuffles' standard deviation
   beside it). The attention measure from round 1 is kept,
   computed over the verified audio positions, for all
   layers, on the round 1 subset shape (first 6 tone and
   first 6 silence clips).

Discipline carried over from round 1: the model loads in
eager attention (the fast mode returns no attention
numbers), the seed is fixed and printed, a check step fails
loudly, full candidate tracebacks go to a readable load log
in the Drive folder, outputs land in the Drive folder
tracing-qwen3b, and the features file is saved as float16
with its size reported.

Plain language throughout, no em dashes.
"""
import copy
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_NB = os.path.join(HERE, "project_k_qwen_omni_colab.ipynb")

# ------------------------------------------------------------ battery spec
# The counts and the CSV schemas live here as literals so
# the build can be verified without running anything.
SEED = 20261105
N_TONE = 360
N_SPEECH = 0
N_SILENCE = 60
N_TRAIN_PER_CLASS = 240
N_TEST_PER_CLASS = 120
N_SHUFFLES = 5
PROBE_C = 0.01
PCA_COMPONENTS = 64
MIN_AUDIO_POSITIONS = 10
PROBE_CSV_FIELDS = ["state", "task", "readout", "n_train", "n_test",
                    "accuracy", "shuffled_accuracy_mean",
                    "shuffled_accuracy_std", "shuffled_p95"]
ATTENTION_CSV_FIELDS = ["clip", "layer", "attention_to_audio"]
MANIFEST_FIELDS = ["clip_name", "group", "split", "file", "freq_hz",
                   "start_s", "burst_s", "level_dbfs", "gain_db",
                   "speed_factor", "floor_dbfs", "rms_dbfs",
                   "wave_sha256", "prompt", "pitch_band", "onset_band"]
assert N_TONE + N_SPEECH + N_SILENCE == 420


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip().splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip().splitlines(True)}


with open(BASE_NB) as fh:
    base_nb = json.load(fh)

# Step 1 is the Series One Qwen notebook's own, reused the
# way the round 1 tracing builder reused it: same session
# installs, same pins, plus the stack proof block
# (transformers exactly 4.52.4) and the scikit-learn proof
# the probes need. Byte-identical to round 1's Step 1.
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
# Project K, Series Three step 5: the property probes, Qwen2.5-Omni-3B

**What this notebook is.** The Handover Correction's last
step (Nov 3, 2026). Round 2's presence helpers saturated:
1.00 at every state, no dynamic range, and a silence class
that leaked across its split. This notebook asks a finer
question with a harder instrument. The sounds are tones
that vary on two factors at once: pitch (500 vs 1500 Hz)
and onset (0.5 vs 2.5 s), 90 unique instances per cell,
every waveform a distinct identity, split by instance
inside each cell. The helpers now read PROPERTIES (high vs
low pitch; late vs early onset), at two positions (the
sound slots pooled, and the final answer position), with
200-permutation nulls in the focus band (states 16 to 25,
decoder layers 15 to 24). A presence helper rides along as
the sanity check, with round 2's 5-shuffle controls. If a
property reads well above its null in the band where the
surgery found the handover, the property survives in
readable form there; if it sits at its null, presence was
all that ever survived. Everything below about round 2 is
the machinery's origin, kept as context; this paragraph
governs what runs.

**Why this notebook exists.** Round 1 of the tracing walked
the sound through Qwen 3B's layers with per-layer helpers
(tiny classifiers trained on one layer's numbers at a time)
and an attention watch. The run finished clean, and then
its own control voided it: a helper trained on shuffled
labels, where there is nothing real to learn, sometimes
scored perfectly too. With 64 training clips against a
2,048-number layer vector, the helpers could memorize
shapes instead of reading the sound. Round 1 also left a
question open: its sanity gate found only 1 audio token in
a 287-position row, where this model family normally
spreads audio across many positions. Round 2 fixes the
probe design. The method, the question, and the registered
predictions are unchanged from the tracing plan.

**The goal.** Find the layer where the loud tone fades
inside the model, and test whether the speech is ever lost
at all. Round 2 carries the helpers and the attention
watch. Surgery (activation patching) is still the follow-up
build, and still the causal check on whatever this round
finds.

**What changed, exactly three things.**

1. **The token check comes first.** Before any extraction,
   one tone clip and one speech clip go through the
   processor, and the runner prints the full histogram of
   token ids in each processed row: every id that occurs,
   its count, and its decoded name. The positions the
   notebook pools over are derived from what the histogram
   shows: the audio candidate id that occurs at
   audio-frame scale, not the first candidate that merely
   occurs once (that is what round 1's gate settled for).
   If the derived audio-position count is under 10 for a
   clip, the step prints TOKEN CHECK FAILED and stops the
   run before anything is extracted. The transcript is
   saved as `property_token_check.txt`.
2. **A corrected battery.** 420 clips: 360 factorial tones, 60
   class (tone, speech, silence), each 4.0 seconds,
   generated inside the runner with a fixed seed and fed
   as raw WAV, no compression anywhere. Inside each class
   the clips vary, so a helper must read the sound rather
   than memorize one waveform. Every clip is measured and
   printed before use: per-second RMS and a waveform hash,
   the feed proof.
3. **A smaller, regularized readout.** Per layer, the
   vectors at the verified audio positions are averaged.
   Two helpers read each layer: a logistic regression with
   strong regularization (C=0.01) on standardized features,
   and the same probe after the features are reduced to 64
   principal components. Each helper trains on 240 clips
   per class and is tested on 120 held-out clips per
   class, the same split at every layer. Beside every real
   accuracy, the results carry the shuffled-label control
   as the mean of 5 shuffles trained and tested
   identically.

**The helpers, in plain words.** A helper is a tiny
classifier we train ourselves, on one layer's numbers only.
We play clips we chose, so we always know which clips carry
the sound and which do not. For each layer, the helper
learns the shape that layer's numbers take when the sound
is in them, then walks the layers on clips it never saw in
training. The layer where the helper stops finding the
sound is the neighborhood where the sound faded. The helper
only reads; it never changes the model, and the model is
never taught anything.

**Readout rules, fixed in advance** (from the tracing
plan, unchanged):

- A helper failing at a layer means "not readable by this
  helper," which can also mean the information changed
  shape. No stage is declared lost on the helpers alone;
  the surgery build is the causal check.
- The shuffled-label control must sit at chance (about
  0.50). A control that beats chance means the helper is
  cheating on some side pattern, and the run is void until
  fixed.
- Predictions, registered before round 1 and unchanged:
  the tone stays readable into the middle layers, then
  fades; the attention the answer pays to the tone's
  tokens is near zero; the speech stays readable at every
  layer, because its transcription is perfect, so its
  denial lives in how the answer is formed, past the
  information.

**Where it runs.** On the free Colab account (kadjei1),
under the routing rule: the 3B is a small model. **How to
run it.** Run the cells from the top, in order. Step 1 sets
up and proves the stack. Step 2 gets the two master sounds.
Step 3 writes the runner and runs its check mode (proof,
loading candidates, the token check). Step 4 runs battery
mode: clip generation, extraction, the helpers, the
attention readout, and a plain-language tail.

**The one rule.** We claim only what we ran: these sounds,
this model, these layers, this run.
"""))
cells.append(step1_md)
cells.append(step1_code)

# ---------------------------------------------------------------- step 2
cells.append(md(r"""
## Step 2 · Get the two master sounds

Two files from your Google Drive folder named
`codec-control` anchor this run:

- `codec_tone_ref_raw.wav`: the loud 1000 Hz tone master,
  uncompressed. The token check's tone clip.
- `codec_speech_raw.wav`: the probe sentence master, the
  swap study's own speech track, uncompressed. The token
  check's speech clip, and the source every speech clip in
  the battery is rendered from.

The next cell links your Drive, finds that folder, and
copies both files to a plain work folder here. The runner
re-measures everything from inside the session before it
believes any of it.

Do not go on until both files say "ready". If a file is
missing, the cell stops and names it.
"""))

cells.append(code(r"""
# --- Get the two master sounds ---------------------------------------------------
import os, shutil, subprocess

FILES = ["codec_tone_ref_raw.wav", "codec_speech_raw.wav"]
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
print("ready: both masters (tone, speech)")
"""))

# ------------------------------------------------------------- runner script
# The runner is a plain Python script the notebook writes to
# /content/qwen_property_run.py and runs under the notebook
# session's own Python, the Qwen notebooks' own pattern.
RUNNER_SRC = r"""
# qwen_property_run.py - the tracing round 2 runner for
# Qwen2.5-Omni-3B. Written by
# project_k_qwen_property_colab.ipynb. It runs under the
# notebook session's own Python: the Qwen notebooks install
# their stack into the session (Step 1), and there is no
# private environment to run under. Modes: "check" runs the
# proof, the loading candidates, and the TOKEN CHECK, then
# stops. "battery" does all of that and then generates the
# probe clips, extracts the per-layer features, trains the
# helpers (two readouts, 5-shuffle controls), reads the
# attention subset, and saves every output.
#
# Exit codes, one per failure family, all loud:
#   3  the stack proof failed (transformers is not 4.52.4)
#   4  both loading candidates failed
#   7  a feed returned no audio (never padded, never scored)
#   9  probe clip generation failed its own measurements
#   10 the extraction count does not match the manifest count
#   11 the token check failed: the audio positions could
#      not be settled from the histogram, or their count is
#      implausible, or the thinker returned no hidden states
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

LOAD_LOG_PATHS = out_paths("property_load_log.txt")

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
log_load("Qwen2.5-Omni-3B tracing round 2 load log - "
         + datetime.datetime.now().isoformat(timespec="seconds")
         + " - runner mode: " + MODE + "\n")

# --- Load the model: the codec runner's candidates ---------------------------------
# Same candidates, same order, same talker-off call as
# round 1. The one printed difference is
# attn_implementation="eager" (see the proof lines above
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
# Sound alone, the codec control's raw arm form: the
# conversation carries an audio element naming the WAV file
# and a text element, and no image and no video element.
# process_mm_info decodes the audio element with the same
# librosa call at 16 kHz mono it uses for a video
# soundtrack. The text is the Series Two presence question
# for the clip's sound, fixed per class, byte-identical to
# round 1: the tracing walks the model's insides while it
# hears the clip under the same question the outside tests
# asked. Silence clips carry the tone question, the primary
# task's question, exactly as round 1's negatives did, and
# the manifest records the exact prompt on every clip.
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
    "silence": ("Is there the sound of a tone in this clip? "
                "Answer yes or no."),
}

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
    # (a generated true-silence clip): an all-zeros feed on
    # a clip that should carry sound is a broken feed, not a
    # quiet clip, and scoring it anyway is how silent
    # failures get into the record.
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
            f"that should carry sound (decoded length "
            f"{len(audio_np)}, decoded rms "
            f"{rms_db_np(audio_np):.2f} dB)")
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
"""

RUNNER_SRC += r"""
# --- Run log: every print also lands in a file in the Drive
# folder, so a stopped run's evidence never depends on the
# notebook output area's scrolling.
class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
            except Exception:
                pass
        return len(s)
    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass
try:
    _log_fh = open("/content/drive/MyDrive/tracing-qwen3b/"
                   "property_run_log.txt", "w", buffering=1)
except Exception:
    _log_fh = open("/content/property_run_log.txt", "w",
                   buffering=1)
sys.stdout = _Tee(sys.stdout, _log_fh)

# --- The token check -----------------------------------------------------------------
# Round 2 fix 1. Round 1's gate picked the first candidate
# id that decoded to the audio pad token and occurred at
# least once, and it occurred exactly once: the probes may
# have pooled over a single marker position instead of the
# audio positions. This check settles the question from the
# data, in the open, before any extraction:
#
#   * for one tone clip and one speech clip (the masters),
#     print the FULL histogram of token ids in the processed
#     input row: every id with count > 0, with the token's
#     name and its decoded form, special tokens flagged;
#   * print the processor's audio feature lengths (the
#     input_features shape and the feature_attention_mask
#     sums), the independent record of how many audio frames
#     the processor built;
#   * collect the audio-token candidates exactly as round 1
#     did (tokenizer tokens whose text mentions audio, the
#     config attributes, the converted pad id), print each
#     candidate's count in the row, and DERIVE the pooling
#     id from the histogram: the candidate with the highest
#     count, the one that occurs at audio-frame scale;
#   * both clips must derive the same id, and its count
#     must be at least MIN_AUDIO_POSITIONS, or the step
#     prints TOKEN CHECK FAILED and stops the run (exit 11)
#     before any extraction;
#   * one thinker forward on the tone clip then confirms
#     the hidden states come back and fixes the layer count
#     and hidden size the extraction will use.
# Every line of the check is also collected and saved to
# property_token_check.txt, pass or fail, so the histogram
# is in the record either way.
from collections import Counter

MIN_AUDIO_POSITIONS = __MIN_AUDIO_POSITIONS__
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
    save_both("property_token_check.txt", w)

def token_check_fail(reason):
    tc()
    tc("!" * 70)
    tc("TOKEN CHECK FAILED: " + reason)
    tc("The pooling positions are not settled, so no extraction")
    tc("has run and no later number in this run can be trusted.")
    tc("The full histogram above is the evidence. Show this whole")
    tc("output to Muse.")
    tc("!" * 70)
    save_token_check()
    sys.exit(11)

def audio_candidates():
    # The candidate audio ids, from the model's own
    # tokenizer and config, never a guessed number. Same
    # evidence round 1 gathered; what changed is how the
    # winner is picked (see token_check).
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
    # Feed one master clip and print its full token
    # histogram plus the processor's audio feature lengths.
    # Returns (inputs, chosen_id, chosen_count, positions).
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
    return inputs, chosen, cand_counts[chosen], positions

def token_check():
    global AUDIO_TOKEN_ID, N_STATES, HIDDEN_SIZE
    tc("=" * 70)
    tc("TOKEN CHECK: settle the audio positions from the "
       "histogram, before any extraction")
    tc(f"plausibility floor: the derived audio-position count "
       f"must be at least {MIN_AUDIO_POSITIONS}")
    candidates = audio_candidates()
    tone_path = os.path.join(CLIPS, "codec_tone_ref_raw.wav")
    speech_path = os.path.join(CLIPS, "codec_speech_raw.wav")
    tone_inputs, tone_id, tone_count, _ = histogram_for(
        "tone", tone_path, PROMPTS["tone"], candidates)
    _, speech_id, speech_count, _ = histogram_for(
        "speech", speech_path, PROMPTS["speech"], candidates)
    if tone_id != speech_id:
        token_check_fail(
            f"the two clips derive different audio token ids "
            f"(tone {tone_id}, speech {speech_id}); the pooling "
            f"positions cannot be settled on a split answer")
    if tone_count < MIN_AUDIO_POSITIONS \
            or speech_count < MIN_AUDIO_POSITIONS:
        token_check_fail(
            f"the derived audio-position count is implausible "
            f"(tone {tone_count}, speech {speech_count}, floor "
            f"{MIN_AUDIO_POSITIONS}); the histogram does not "
            f"show an id occurring at audio-frame scale")
    AUDIO_TOKEN_ID = tone_id
    tc("-" * 70)
    tc(f"AUDIO TOKEN ID SETTLED: {AUDIO_TOKEN_ID}, from the "
       f"histogram (tone count {tone_count}, speech count "
       f"{speech_count})")
    out = thinker_forward(tone_inputs)
    hs = getattr(out, "hidden_states", None)
    if hs is None or len(hs) == 0:
        token_check_fail(
            "the thinker returned no hidden states on the tone "
            "clip; there would be nothing to pool")
    N_STATES = len(hs)
    HIDDEN_SIZE = int(hs[0].shape[2])
    tc(f"hidden states returned: {N_STATES} tensors (layer 0 is "
       f"the embedding output, then one per thinker layer)")
    tc(f"thinker layers: {N_STATES - 1}")
    tc(f"hidden size: {HIDDEN_SIZE}")
    tc(f"tone sequence length: {int(hs[0].shape[1])}")
    del out, tone_inputs
    free_memory()
    tc("TOKEN CHECK PASSED: the audio positions are derived "
       "from the histogram, counted, and the same for both "
       "clips. Every later number in this run pools over "
       "these positions.")
    save_token_check()

token_check()

def audio_positions(input_ids_row):
    row = [int(x) for x in input_ids_row]
    return [i for i, x in enumerate(row) if x == AUDIO_TOKEN_ID]

if MODE == "check":
    print()
    print("check mode done: proof, candidates, and the token check")
    print("all passed. No clips were generated and no extraction ran.")
    print("Run Step 4 for the battery.")
    sys.exit(0)
"""

RUNNER_SRC += r"""
# --- Probe clip generation -------------------------------------------------------------
# Round 2 fix 2. Fixed seed, so the run can be repeated
# exactly and the split can be re-derived. 420 clips,
# 360 per class (tone, speech, silence), each exactly
# 4.0 s at 16 kHz mono (64000 samples), fed later as raw
# WAV: no MP4, no codec anywhere.
#
# Variation inside each class, so a helper must read the
# sound and not memorize one waveform:
#   tone: a 0.5 s sine burst, frequency jittered 950 to
#     1050 Hz, start 0.5 to 3.0 s, level -24 to -14 dBFS.
#   speech: the probe sentence from the master, tape-style
#     speed factor 0.92 to 1.08 (the resample shifts pitch
#     with speed, the small speed/pitch variation), gain
#     -6 to +3 dB, start offset up to 1.5 s.
#   silence: nothing in the foreground at all.
# Every clip, all classes alike, sits on a background drawn
# from one shared scheme: plain digital silence, with a
# one-in-six chance of a near-silence white-noise floor at
# -75 to -65 dBFS. The silence class is therefore true
# digital silence plus a few near-silence noise-floor
# variants, and the background can never be a class cue,
# because every class draws from the same scheme.
#
# The split is fixed by seed: 240 train / 120 held-out test
# per class, stratified, the same split for every layer and
# both tasks. split_ranks below is the single source of
# truth: generation stamps each clip's split in the
# manifest, and the probe step re-derives the same ranks and
# asserts the two agree.
#
# Every clip is measured before anything uses it (sample
# count, whole-clip RMS, per-second RMS, waveform SHA-256)
# and printed: the feed proof. A failed measurement stops
# the run (exit 9).
import csv, hashlib

SEED = __SEED__
N_TRAIN = __N_TRAIN__
SR = 16000
CLIP_SAMPLES = SR * 4
CLASS_NUMBER = {"tone": 1, "speech": 2, "silence": 3}
CLIP_DIRS = ["/content/drive/MyDrive/tracing-qwen3b/property_clips",
             "/content/property_clips"]
for d in CLIP_DIRS:
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
MANIFEST_FIELDS = __MANIFEST_FIELDS__

def split_ranks(class_name):
    # The within-class ranks (0-based, in generation order)
    # that belong to train and to test for one class.
    rng = np.random.default_rng(SEED + 1009 * CLASS_NUMBER[class_name])
    order = rng.permutation(360)
    return (set(int(x) for x in order[:N_TRAIN]),
            set(int(x) for x in order[N_TRAIN:]))

def rms_db_np(x):
    if len(x) == 0:
        return -120.0
    v = float(np.sqrt(np.mean(np.square(x.astype(np.float64)))))
    return 20.0 * math.log10(v) if v > 0 else -120.0

def db_to_amp(db):
    return float(10.0 ** (db / 20.0))

def wave_hash(wave):
    return hashlib.sha256(
        np.asarray(wave, dtype=np.float32).tobytes()).hexdigest()

def background(rng):
    # One draw from the shared background scheme: (wave,
    # floor level in dBFS or None for digital silence).
    wave = np.zeros(CLIP_SAMPLES, dtype=np.float32)
    if rng.random() < 1.0 / 6.0:
        level = float(rng.uniform(-75.0, -65.0))
        noise = rng.standard_normal(CLIP_SAMPLES).astype(np.float32)
        cur = float(np.sqrt(np.mean(np.square(
            noise.astype(np.float64)))))
        wave = (noise * (db_to_amp(level) / cur)).astype(np.float32)
        return wave, round(level, 2)
    return wave, None

def write_clip(name, wave):
    wave = np.asarray(wave, dtype=np.float32)
    assert len(wave) == CLIP_SAMPLES, \
        f"clip {name} has {len(wave)} samples, not {CLIP_SAMPLES}"
    local = os.path.join("/content/property_clips", name)
    drive = os.path.join(
        "/content/drive/MyDrive/tracing-qwen3b/property_clips", name)
    # ALWAYS write the wave in hand, to both homes, overwriting
    # whatever is there. Never copy an existing file into place:
    # /content is fresh each session but the Drive folder
    # persists across runs, and an earlier run's file under the
    # same name can hold a different wave (the draw sequence
    # moved when the builder changed). That stale-file reuse is
    # exactly what stopped the Nov 3 diagnostic run at clip
    # 413: prop_silence_53's row described a -67.5 dB noise
    # floor while the file on disk was an older run's digital
    # silence. The read-back check in finish_row now proves
    # the file on disk is the wave in hand before extraction
    # ever sees it.
    sf.write(local, wave, SR, subtype="PCM_16")
    try:
        sf.write(drive, wave, SR, subtype="PCM_16")
    except Exception:
        pass
    return local

def speech_segment():
    # The sentence inside the speech master: the region whose
    # short-window RMS rises above -45 dBFS, with a small pad.
    # Same cut as round 1.
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

def tape_resample(x, speed):
    # Tape-style speed change: playing the segment faster or
    # slower, which shifts its pitch by the same factor. The
    # round 2 speech variation (speed 0.92 to 1.08, about a
    # semitone and a half either way at the extremes).
    n_out = max(1, int(round(len(x) / speed)))
    idx = np.linspace(0, len(x) - 1, n_out)
    return np.interp(idx, np.arange(len(x)), x).astype(np.float32)

def finish_row(rows, name, group, rank, wave, prompt, extra):
    # Peak guard, split stamp, measurement, and the printed
    # feed proof for one generated clip.
    peak = float(np.max(np.abs(wave))) if len(wave) else 0.0
    if peak > 0.99:
        wave = (wave * (0.99 / peak)).astype(np.float32)
        print(f"  {name}: peak {peak:.2f} over the guard; scaled "
              f"to a 0.99 peak")
    path = write_clip(name, wave)
    # Read-back proof: the file on disk must be the wave in
    # hand (PCM_16 quantisation moves the RMS by far less
    # than 1 dB at these levels; digital silence reads back
    # at exactly -120). A mismatch stops generation here,
    # where the cause is visible, instead of surfacing as a
    # feed hard stop deep inside extraction.
    _back, _ = sf.read(path, dtype="float32")
    if len(_back) != len(wave) or abs(rms_db_np(_back)
                                      - rms_db_np(wave)) > 1.0:
        print()
        print("!" * 70)
        print(f"CLIP WRITE CHECK FAILED: {name} on disk does "
              f"not match the generated wave (disk rms "
              f"{rms_db_np(_back):.2f} dB vs wave rms "
              f"{rms_db_np(wave):.2f} dB). Nothing was "
              f"extracted. Show this whole output to Muse.")
        print("!" * 70)
        sys.exit(9)
    train_ranks, _ = split_ranks(group)
    split = "train" if rank in train_ranks else "test"
    row = {"clip_name": name[:-4], "group": group, "split": split,
           "file": path, "wave_sha256": wave_hash(wave),
           "rms_dbfs": round(rms_db_np(wave), 2), "prompt": prompt,
           "_expect_silence": bool(group == "silence"
                                   and extra.get("floor_dbfs") in ("", None))}
    row.update(extra)
    split = row["split"]
    sec = [rms_db_np(wave[k * SR:(k + 1) * SR]) for k in range(4)]
    print(f"  {row['clip_name']:<14} group={group:<7} split={split:<5} "
          f"samples={len(wave)} rms={row['rms_dbfs']:.2f} dB "
          f"per-second [{', '.join(f'{v:.1f}' for v in sec)}] dB "
          f"sha256={row['wave_sha256'][:16]}")
    if len(wave) != CLIP_SAMPLES:
        print()
        print("!" * 70)
        print(f"CLIP MEASUREMENT FAILED: {row['clip_name']} has "
              f"{len(wave)} samples, not {CLIP_SAMPLES}. The "
              f"generated set is not what the plan says. Nothing "
              f"was extracted. Show this whole output to Muse.")
        print("!" * 70)
        sys.exit(9)
    rows.append(row)

def generate_clips():
    # The Handover Correction step 5 stimulus set. Factorial
    # tones: pitch (500 vs 1500 Hz) crossed with onset (0.5 vs
    # 2.5 s), 90 instances per cell, 360 tones; 60 silence
    # clips (the shared background scheme alone). Every
    # instance is a fresh draw (pitch jitter x0.98 to 1.02,
    # onset jitter +-0.1 s, level -20 +-1 dBFS, random phase),
    # so every waveform is a unique identity; the split is by
    # instance within each cell (60 train / 30 test), stamped
    # here and asserted unique by hash below.
    rng = np.random.default_rng(SEED)
    rows = []
    cells = [("low", 500.0, "early", 0.5), ("low", 500.0, "late", 2.5),
             ("high", 1500.0, "early", 0.5), ("high", 1500.0, "late", 2.5)]
    for ci, (pband, pbase, oband, obase) in enumerate(cells):
        order = rng.permutation(90)
        for i in range(90):
            freq = float(pbase * rng.uniform(0.98, 1.02))
            start_s = float(obase + rng.uniform(-0.1, 0.1))
            level = float(-20.0 + rng.uniform(-1.0, 1.0))
            phase = float(rng.uniform(0.0, 2 * np.pi))
            burst_s = 0.5
            wave, floor_db = background(rng)
            n_burst = int(burst_s * SR)
            t = np.arange(n_burst) / SR
            burst = (db_to_amp(level)
                     * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)
            edge = SR // 200
            ramp = 0.5 - 0.5 * np.cos(
                np.linspace(0, np.pi, edge)).astype(np.float32)
            burst[:edge] *= ramp
            burst[-edge:] *= ramp[::-1]
            s0 = int(start_s * SR)
            wave[s0:s0 + n_burst] += burst
            split = "train" if int(order[i]) < 60 else "test"
            finish_row(rows, f"prop_tone_{ci}_{i:02d}.wav", "tone", i, wave,
                       PROMPTS["tone"],
                       {"freq_hz": round(freq, 1),
                        "start_s": round(start_s, 3), "burst_s": burst_s,
                        "level_dbfs": round(level, 2), "gain_db": "",
                        "speed_factor": "",
                        "floor_dbfs": floor_db if floor_db is not None else "",
                        "pitch_band": pband, "onset_band": oband,
                        "split": split})
    # Silence: the shared background scheme returns pure
    # digital silence for most draws, so identical silence
    # waveforms are expected and legitimate (silence is one
    # stimulus). The discipline that matters is split
    # hygiene: identical waveforms are grouped and a group
    # sits entirely on one side of the split, so no stimulus
    # is ever on both sides. Draw all 60 first, group by
    # hash, then stamp.
    sil_draws = []
    for i in range(60):
        wave, floor_db = background(rng)
        sil_draws.append((i, wave, floor_db))
    groups = {}
    for i, wave, floor_db in sil_draws:
        groups.setdefault(wave_hash(wave), []).append((i, wave, floor_db))
    gorder = list(groups.keys())
    rng.shuffle(gorder)
    train_hashes = set()
    n_train = 0
    for h in gorder:
        if n_train >= 40:
            break
        train_hashes.add(h)
        n_train += len(groups[h])
    for h in gorder:
        split = "train" if h in train_hashes else "test"
        for i, wave, floor_db in groups[h]:
            finish_row(rows, f"prop_silence_{i:02d}.wav", "silence", i, wave,
                       PROMPTS["silence"],
                       {"freq_hz": "", "start_s": "", "burst_s": "",
                        "level_dbfs": "", "gain_db": "", "speed_factor": "",
                        "floor_dbfs": floor_db if floor_db is not None else "",
                        "pitch_band": "", "onset_band": "",
                        "split": split})
    print("-" * 70)
    tone_hashes = [r["wave_sha256"] for r in rows if r["group"] == "tone"]
    assert len(set(tone_hashes)) == len(tone_hashes), \
        "duplicate tone waveform hashes; the property identities must be unique"
    sil_by_split = {}
    for r in rows:
        if r["group"] == "silence":
            sil_by_split.setdefault(r["wave_sha256"], set()).add(r["split"])
    assert all(len(v) == 1 for v in sil_by_split.values()), \
        "a silence waveform sits on both sides of the split"
    assert not (set(tone_hashes) & set(sil_by_split)), \
        "a tone waveform hash collides with a silence waveform"
    print(f"generated {len(rows)} clips; {len(tone_hashes)} tone waveforms, "
          f"all unique; silence in {len(sil_by_split)} unique waveforms, "
          f"each entirely on one side of the split")
    from collections import Counter
    splits = Counter((r["group"], r["split"]) for r in rows)
    print(f"  tone: {splits[('tone', 'train')]} train, {splits[('tone', 'test')]} test")
    print(f"  silence: {splits[('silence', 'train')]} train, {splits[('silence', 'test')]} test")
    def write_manifest(p):
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in MANIFEST_FIELDS})
    save_both("property_manifest.csv", write_manifest)
    return rows
"""

RUNNER_SRC += r"""
# --- Extraction --------------------------------------------------------------------------
# One frozen forward pass per clip. Per layer, the vectors
# at the VERIFIED audio positions (the token check's id)
# are averaged into one vector per clip and stored. The
# attention subset (the first 6 tone and first 6 silence
# clips, in manifest order, the round 1 subset shape) gets
# a second forward with attentions returned; for EVERY
# layer we record the mean attention from the final sequence
# position to the verified audio positions. Features are
# saved as float16 (property_features.npz) with the file
# size printed.
FEATURES_NAME = "property_features.npz"

def find_features():
    for p in out_paths(FEATURES_NAME):
        if os.path.exists(p):
            return p
    return None

def extract(rows):
    n_clips = len(rows)
    feats = np.zeros((N_STATES, n_clips, HIDDEN_SIZE),
                     dtype=np.float32)
    feats_final = np.zeros((N_STATES, n_clips, HIDDEN_SIZE),
                           dtype=np.float32)
    audio_counts = np.zeros(n_clips, dtype=np.int64)
    seq_lens = np.zeros(n_clips, dtype=np.int64)
    attention_rows = []
    subset = set()
    tone_seen = sil_seen = 0
    for idx, r in enumerate(rows):
        if r["group"] == "tone" and tone_seen < 6:
            subset.add(idx)
            tone_seen += 1
        elif r["group"] == "silence" and sil_seen < 6:
            subset.add(idx)
            sil_seen += 1
    for idx, r in enumerate(rows):
        expect_silence = bool(r.get("_expect_silence", False))
        try:
            inputs, audio_np = feed_inputs(
                r["file"], r["prompt"],
                expect_silence=expect_silence)
        except FeedError as e:
            print()
            print("!" * 70)
            print("EXTRACTION FEED HARD STOP:")
            print(str(e))
            try:
                _w, _sr = sf.read(r["file"], dtype="float32")
                print(f"  on-disk check for {r['file']}: "
                      f"{len(_w)} samples at {_sr} Hz, rms "
                      f"{rms_db_np(np.asarray(_w)):.2f} dB, "
                      f"any nonzero: {bool(np.any(_w != 0))}, "
                      f"expect_silence flag: {expect_silence}")
            except Exception as _e2:
                print(f"  on-disk check failed: {str(_e2)[:160]}")
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
                  f"zero audio-token positions under the token "
                  f"check's derived id {AUDIO_TOKEN_ID}. The check "
                  f"found them on both masters, so this clip's feed "
                  f"differs somehow. No vector was faked. Show this "
                  f"output to Muse.")
            print("!" * 70)
            sys.exit(11)
        out = thinker_forward(inputs)
        hs = out.hidden_states
        pos_t = torch.tensor(pos, dtype=torch.long)
        for li in range(N_STATES):
            layer_vec = hs[li][0].index_select(0, pos_t.to(
                hs[li].device)).mean(dim=0)
            feats[li, idx] = layer_vec.float().cpu().numpy()
            feats_final[li, idx] = hs[li][0][-1].float().cpu().numpy()
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
                for li in range(len(atts)):
                    a = atts[li][0].float()  # heads, seq, seq
                    val = float(a[:, -1, :][:, pos].mean().item())
                    attention_rows.append({
                        "clip": r["clip_name"], "layer": li,
                        "attention_to_audio": round(val, 6)})
                print(f"attention read: {r['clip_name']} "
                      f"(all {len(atts)} layers)")
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
        np.savez(p, features=feats.astype(np.float16),
                 features_final=feats_final.astype(np.float16),
                 groups=groups, clip_names=names,
                 audio_token_counts=audio_counts, seq_lens=seq_lens)
    saved = save_both(FEATURES_NAME, write_npz)
    for p in saved:
        print(f"features file (float16): {p}, "
              f"{os.path.getsize(p) / 1e6:.1f} MB")
    return feats, feats_final, groups, names, attention_rows

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
            feats = f0.astype(np.float32)
            groups, names = z["groups"], z["clip_names"]
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
    feats, feats_final, groups, names, attention_rows = extract(rows)

if attention_rows:
    def write_attention(p):
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=__ATTENTION_FIELDS__)
            w.writeheader()
            w.writerows(attention_rows)
    save_both("property_attention.csv", write_attention)

# --- The helpers (probes), round 2 readout -----------------------------------------------------
# Round 2 fix 3. Per layer, one vector per clip: the mean
# over the verified audio positions. Two readouts per layer
# per task, both small on purpose:
#   plain: standardized features, logistic regression with
#     strong regularization (C=0.01);
#   pca64: standardized features reduced to 64 principal
#     components, then the same probe.
# Tasks unchanged from round 1: tone vs silence, speech vs
# silence. The split is the seed-fixed split from
# generation: 240 train / 120 test per class, the same at
# every layer, re-derived here from split_ranks and asserted
# against the manifest stamps. Every helper is shadowed by
# a shuffled-label control: 5 shuffles of the training
# labels, each trained and tested identically to the real
# helper, reported as their mean (and standard deviation).
# A control that beats chance voids the run under the plan.
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

PROBE_FIELDS = __PROBE_FIELDS__
N_SHUFFLES = __N_SHUFFLES__
PROBE_C = __PROBE_C__
PCA_COMPONENTS = __PCA_COMPONENTS__
group_list = [str(g) for g in groups]

# Splits and labels come from the manifest stamps written at
# generation: per-cell instance splits for the tones, so no
# waveform identity appears on both sides of any task.
tone_tr = [i for i, r in enumerate(rows)
           if r["group"] == "tone" and r["split"] == "train"]
tone_te = [i for i, r in enumerate(rows)
           if r["group"] == "tone" and r["split"] == "test"]
sil_tr = [i for i, r in enumerate(rows)
          if r["group"] == "silence" and r["split"] == "train"]
sil_te = [i for i, r in enumerate(rows)
          if r["group"] == "silence" and r["split"] == "test"]
print(f"tone split: {len(tone_tr)} train, {len(tone_te)} test; "
      f"silence split: {len(sil_tr)} train, {len(sil_te)} test")

def band_labels(idxs, key, positive):
    return np.array([1 if rows[i][key] == positive else 0
                     for i in idxs])

TASKS = {
    "pitch_high_vs_low": (
        tone_tr, tone_te,
        band_labels(tone_tr, "pitch_band", "high"),
        band_labels(tone_te, "pitch_band", "high")),
    "onset_late_vs_early": (
        tone_tr, tone_te,
        band_labels(tone_tr, "onset_band", "late"),
        band_labels(tone_te, "onset_band", "late")),
    "presence_tone_vs_silence": (
        tone_tr + sil_tr, tone_te + sil_te,
        np.array([1] * len(tone_tr) + [0] * len(sil_tr)),
        np.array([1] * len(tone_te) + [0] * len(sil_te))),
}
READOUTS = {"audio_plain": (feats, "plain"),
            "audio_pca64": (feats, "pca64"),
            "final_plain": (feats_final, "plain"),
            "final_pca64": (feats_final, "pca64")}

def fit_readout(X_train, y_train, readout):
    if readout == "plain":
        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=PROBE_C, max_iter=1000))
    elif readout == "pca64":
        pipe = make_pipeline(
            StandardScaler(),
            PCA(n_components=PCA_COMPONENTS, random_state=SEED),
            LogisticRegression(C=PROBE_C, max_iter=1000))
    else:
        raise ValueError(f"unknown readout {readout}")
    pipe.fit(X_train, y_train)
    return pipe

probe_rows = []
for task, (tr_idx, te_idx, y_train, y_test) in TASKS.items():
    for rname, (F, rtype) in READOUTS.items():
        if task == "presence_tone_vs_silence" and rname.startswith("final"):
            continue
        for li in range(N_STATES):
            X = F[li]
            clf = fit_readout(X[tr_idx], y_train, rtype)
            acc = float(clf.score(X[te_idx], y_test))
            probe_rows.append({
                "state": li, "task": task, "readout": rname,
                "n_train": len(tr_idx), "n_test": len(te_idx),
                "accuracy": round(acc, 4),
                "shuffled_accuracy_mean": "",
                "shuffled_accuracy_std": "",
                "shuffled_p95": ""})
            print(f"probe {task} {rname} state {li}: "
                  f"accuracy {acc:.3f}")

def write_probes(p):
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=PROBE_FIELDS)
        w.writeheader()
        w.writerows(probe_rows)
save_both("property_probe_results.csv", write_probes)
print("headline probe table saved before the controls run")

# Presence controls: 5 shuffles, as round 2, audio readouts.
tr_idx, te_idx, y_train, y_test = TASKS["presence_tone_vs_silence"]
for rname in ("audio_plain", "audio_pca64"):
    F, rtype = READOUTS[rname]
    for li in range(N_STATES):
        X = F[li]
        shuf = []
        for s in range(N_SHUFFLES):
            rng_s = np.random.default_rng(SEED + 500000 + li * 1000 + s)
            clf_s = fit_readout(X[tr_idx], rng_s.permutation(y_train), rtype)
            shuf.append(float(clf_s.score(X[te_idx], y_test)))
        for r in probe_rows:
            if (r["task"] == "presence_tone_vs_silence"
                    and r["readout"] == rname and r["state"] == li):
                r["shuffled_accuracy_mean"] = round(float(np.mean(shuf)), 4)
                r["shuffled_accuracy_std"] = round(float(np.std(shuf)), 4)
                r["shuffled_p95"] = round(float(np.max(shuf)), 4)
    print(f"presence controls done for {rname} (5 shuffles per state)")

# Property nulls: 200 permutations in the focus band (states
# 16 to 25, the outputs of decoder layers 15 to 24), pca64
# readouts at both positions.
FOCUS_STATES = list(range(16, 26))
N_PERMS = 200
for task in ("pitch_high_vs_low", "onset_late_vs_early"):
    tr_idx, te_idx, y_train, y_test = TASKS[task]
    for rname in ("audio_pca64", "final_pca64"):
        F, rtype = READOUTS[rname]
        for li in FOCUS_STATES:
            X = F[li]
            nulls = []
            for s in range(N_PERMS):
                rng_p = np.random.default_rng(
                    SEED + 900000 + li * 10000 + s * 17
                    + (0 if rname == "audio_pca64" else 5000)
                    + (0 if task.startswith("pitch") else 2500))
                clf_s = fit_readout(X[tr_idx],
                                    rng_p.permutation(y_train), rtype)
                nulls.append(float(clf_s.score(X[te_idx], y_test)))
            for r in probe_rows:
                if (r["task"] == task and r["readout"] == rname
                        and r["state"] == li):
                    r["shuffled_accuracy_mean"] = round(float(np.mean(nulls)), 4)
                    r["shuffled_accuracy_std"] = round(float(np.std(nulls)), 4)
                    r["shuffled_p95"] = round(float(np.percentile(nulls, 95)), 4)
            print(f"nulls {task} {rname} state {li}: mean "
                  f"{np.mean(nulls):.3f}, p95 {np.percentile(nulls, 95):.3f} "
                  f"({N_PERMS} permutations)")
save_both("property_probe_results.csv", write_probes)

# --- The plain-language tail ---------------------------------------------------------------------
print()
print("=" * 70)
print("READOUT (mechanical, no verdict beyond the numbers)")
for task in ("pitch_high_vs_low", "onset_late_vs_early"):
    for rname in ("audio_pca64", "final_pca64"):
        band = [r for r in probe_rows if r["task"] == task
                and r["readout"] == rname and r["state"] in FOCUS_STATES]
        accs = [r["accuracy"] for r in band]
        p95s = [r["shuffled_p95"] for r in band]
        print(f"  {task} [{rname}], states 16 to 25: accuracy "
              f"min {min(accs):.3f}, max {max(accs):.3f}; "
              f"null p95 max {max(p95s):.3f}")
print("  state k is the output of decoder layer k-1 (state 0 is")
print("  the embedding output); the focus band, states 16 to 25,")
print("  is decoder layers 15 to 24. A probe reading a property")
print("  shows the property is present in readable form at that")
print("  state; it does not show the model uses it.")
print("=" * 70)
print("battery done: token check passed, clips generated and")
print("measured, features extracted and counted, property")
print("helpers trained, focus-band permutation nulls run.")
print("Outputs: property_token_check.txt, property_manifest.csv,")
print("property_features.npz (float16), property_probe_results.csv,")
print("property_attention.csv.")
"""

RUNNER_SRC = RUNNER_SRC.replace("__SEED__", str(SEED))
RUNNER_SRC = RUNNER_SRC.replace("__N_TONE__", str(N_TONE))
RUNNER_SRC = RUNNER_SRC.replace("__N_SPEECH__", str(N_SPEECH))
RUNNER_SRC = RUNNER_SRC.replace("__N_SILENCE__", str(N_SILENCE))
RUNNER_SRC = RUNNER_SRC.replace("__N_TRAIN__", str(N_TRAIN_PER_CLASS))
RUNNER_SRC = RUNNER_SRC.replace("__N_SHUFFLES__", str(N_SHUFFLES))
RUNNER_SRC = RUNNER_SRC.replace("__PROBE_C__", str(PROBE_C))
RUNNER_SRC = RUNNER_SRC.replace("__PCA_COMPONENTS__", str(PCA_COMPONENTS))
RUNNER_SRC = RUNNER_SRC.replace("__MIN_AUDIO_POSITIONS__",
                                str(MIN_AUDIO_POSITIONS))
RUNNER_SRC = RUNNER_SRC.replace("__MANIFEST_FIELDS__", repr(MANIFEST_FIELDS))
RUNNER_SRC = RUNNER_SRC.replace("__PROBE_FIELDS__", repr(PROBE_CSV_FIELDS))
RUNNER_SRC = RUNNER_SRC.replace("__ATTENTION_FIELDS__",
                                repr(ATTENTION_CSV_FIELDS))
for _ph in ("__SEED__", "__N_TONE__", "__N_SPEECH__", "__N_SILENCE__",
            "__N_TRAIN__", "__N_SHUFFLES__", "__PROBE_C__",
            "__PCA_COMPONENTS__", "__MIN_AUDIO_POSITIONS__",
            "__MANIFEST_FIELDS__", "__PROBE_FIELDS__",
            "__ATTENTION_FIELDS__"):
    assert _ph not in RUNNER_SRC, f"placeholder survived: {_ph}"

# ---------------------------------------------------------------- step 3
cells.append(md(r"""
## Step 3 · Write the runner, then check: proof, candidates, token check

The model never runs inside this notebook's cells directly.
It runs in a plain script, `/content/qwen_property_run.py`,
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
   `property_load_log.txt` in the tracing folder.
3. **The token check.** One tone clip and one speech clip
   (the masters from Step 2) go through the processor. For
   each, the runner prints the full histogram of token ids
   in the processed row (every id, its count, its decoded
   name, special tokens flagged), the processor's audio
   feature lengths, and every candidate audio id with its
   count. The pooling id is derived from the histogram: the
   candidate that occurs at audio-frame scale. Both clips
   must derive the same id, and its count must be at least
   10, or the step prints TOKEN CHECK FAILED and stops the
   run (exit code 11) before any extraction. One forward
   pass on the tone clip then confirms the hidden states
   come back and fixes the layer count and hidden size.
   The whole transcript is saved as
   `property_token_check.txt`. No clips are generated and
   no extraction runs in check mode.
"""))

cells.append(code(
    "# --- Write the tracing round 2 runner script -------------------------------------\n"
    "# The runner holds the proof, the loading candidates, the\n"
    "# token check, the clip generation, the extraction, the\n"
    "# helpers, and the attention readout. It is written to a\n"
    "# plain file here and only ever run under this session's\n"
    "# own Python, the interpreter Step 1 installed into.\n"
    "RUNNER_SOURCE = " + repr(RUNNER_SRC) + "\n"
    "with open(\"/content/qwen_property_run.py\", \"w\") as fh:\n"
    "    fh.write(RUNNER_SOURCE)\n"
    "print(\"runner written:\", \"/content/qwen_property_run.py\", "
    "len(RUNNER_SOURCE), \"chars\")\n"))

cells.append(code(r"""
# --- Check mode: proof, candidates, token check --------------------------------------
# Runs the runner under this session's own Python and streams
# its output here live. A nonzero exit code means the proof, a
# loading candidate, or the token check failed, and the run
# stops in this cell. The first check run downloads the model
# files once; that takes a while and is normal.
import subprocess, sys
proc = subprocess.run([sys.executable, "/content/qwen_property_run.py", "check"])
print()
print("check exit code:", proc.returncode)
if proc.returncode != 0:
    print("!" * 72)
    print("STOP. THE CHECK RUN FAILED. The runner's own output above says")
    print("where: a proof line, a loading candidate, or the token check.")
    print("Do not run Step 4. Show this output to Muse.")
    print("!" * 72)
    raise SystemExit("STOP: runner check failed")
print("check passed: proof lines right, model loaded, token check")
print("passed with the audio positions derived and counted. Go on to Step 4.")
"""))

# ---------------------------------------------------------------- step 4
cells.append(md(r"""
## Step 4 · Run the tracing battery

The next cell runs the runner in **battery mode**: proof,
candidates, and the token check again, then four stages.

**Clip generation.** 420 clips are synthesized with the
fixed seed 20261104, each exactly 4.0 seconds at 16 kHz:
360 tone (a 0.5 s burst, frequency jittered between 950
and 1050 Hz, start between 0.5 and 3.0 s, level between
-24 and -14 dBFS), 360 speech (the one probe sentence,
"The blue kettle boils at seven forty two.", rendered
from the master, tape-style speed between 0.92 and 1.08, gain
between -6 and +3 dB, start offset up to 1.5 s), and 360
silence (no foreground at all). Every clip sits on the
shared background scheme: digital silence, with a
one-in-six chance of a near-silence noise floor between
-75 and -65 dBFS. Every clip is measured and printed
(sample count, whole-clip RMS, per-second RMS, waveform
SHA-256) before anything uses it; a clip that fails its
measurement stops the run. The split is stamped per clip
in `property_manifest.csv`: 240 train and 120 test per
class, fixed by the seed.

**Extraction.** Each clip goes through the frozen model
once, sound alone, under the Series Two presence question
for its class. Any feed that returns no audio, or decodes
to zeros on a clip that should carry sound, stops the run
on the spot (exit code 7): nothing is padded and nothing
is scored on a broken feed. Per layer, the vectors at the
verified audio positions are averaged into one vector per
clip and saved to `property_features.npz` as float16, with
the file size printed. The extraction count must equal
the manifest count, 420, or the run stops (exit code
10). If the features file already holds all 420 clips
with matching names, extraction is skipped and the run
goes straight to the helpers.

**The helpers.** Per layer and per task (tone vs silence,
speech vs silence), two readouts train on 240 positives
plus 240 negatives and are tested on the held-out 120
plus 120, the same split at every layer: a logistic
regression with strong regularization (C=0.01) on
standardized features, and the same probe after a PCA
reduction to 64 components. Each helper is shadowed by a
shuffled-label control: 5 shuffles, each trained and
tested identically, reported as their mean and standard
deviation. Results land in `property_probe_results.csv`:
one row per layer per task per readout, controls included.

**The attention readout.** The first 6 tone and first 6
silence clips get a second pass with attentions returned.
For every layer, the mean attention from the final
sequence position to the verified audio positions lands
in `property_attention.csv`.

**The tail.** The runner ends with a plain-language
readout: for each task and readout, the first layer where
the helper falls below 0.70 and stays below (or "never"),
and the shuffled controls' range across the whole run.
"""))

cells.append(code(r"""
# --- Battery mode: the tracing round 2 battery -----------------------------------------
# Same runner, same session Python, streamed live. The runner
# saves its outputs itself, to the Drive tracing folder and
# to /content. A nonzero exit code means the token check
# failed, a feed hard-stopped the run, clip generation failed
# its measurements, or the extraction count mismatched (the
# banners above say which).
import subprocess, sys
proc = subprocess.run([sys.executable, "/content/qwen_property_run.py", "battery"])
print()
print("battery exit code:", proc.returncode)
if proc.returncode != 0:
    print("!" * 72)
    print("STOP. THE BATTERY RUN DID NOT FINISH CLEAN. Read the runner's")
    print("output above: the token check failed, a feed proof hard-stopped")
    print("the run, clip generation failed, or the counts mismatched. Show")
    print("it to Muse.")
    print("!" * 72)
    raise SystemExit("STOP: battery run not clean")
print("battery done: token check passed, features extracted and counted,")
print("helpers trained with 5-shuffle controls, attention read, CSVs saved.")
"""))

# ---------------------------------------------------------------- closing
cells.append(md(r"""
## Closing · How to read the answers

Read each task's curve on its own first: one accuracy per
layer per readout, from the embedding output (layer 0) to
the last thinker layer, with the shuffled control's mean
beside it. The token check transcript,
`property_token_check.txt`, is read before any curve: it
is the proof of which positions every number pooled over,
and how many there were.

- **The tone fades somewhere.** The fade layer is the
  neighborhood where this helper stops reading the tone.
  Read it with the attention numbers beside it: if the
  final position's attention to the tone's positions was
  near zero all along, the answer never looked; if
  attention was healthy and the report still missed, it
  looked and dropped. Either way, the fade is a
  readability result, not yet a proof of loss. The surgery
  build is the causal check that confirms or kills the
  claim.
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
  With 240 training clips per class, two regularized
  readouts, and the control reported as the mean of 5
  shuffles beside every score, round 2 is built so this
  verdict, whichever way it lands, can be trusted.
- **Scope.** Every claim stays capped at this model, these
  sounds, this question per class, this run. Pegasus stays
  a black box; MiniCPM and OLA tracing wait until these
  instruments have run end to end here; the picture
  condition is a later round.

When you finish, the files in the `tracing-qwen3b` Drive
folder are the record: the token check transcript, the
manifest, the features, the probe results, and the
attention readout. Send them back, and the tracing gets
written up next to the codec control.
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

out = os.path.join(HERE, "project_k_qwen_property_colab.ipynb")
with open(out, "w") as fh:
    json.dump(nb, fh, indent=1)
print("wrote", out, "cells:", len(cells))
