#!/usr/bin/env python3
"""Build project_k_qwen_series2_colab.ipynb.

Series Two diagnostic notebook for Qwen2.5-Omni-3B. It reuses the
Series One Qwen notebook's own environment and paths, unchanged:
Step 1 is that notebook's session install (same pins:
transformers==4.52.4, accelerate==1.8.1, qwen-omni-utils==0.0.8,
soundfile==0.13.1, librosa==0.11.0, av), with one Series Two
addition, a proof block that prints the imported transformers
version and file and stops the run if the pin is not met, the way
the MiniCPM and OLA Series Two notebooks prove their stacks.
There is no private environment in the Qwen notebooks and none is
invented here: the runner script runs under the notebook
session's own Python, which is the interpreter Step 1 installed
into. Step 3 keeps the Series One loading candidates (the 3B in
float16 first, the 7B AWQ build second), the disable_talker call,
and the processor load, with full tracebacks appended live to
series2_qwen_load_log.txt in the clips folder. The ask path is
the Series One one (same conversation shape, process_mm_info with
use_audio_in_video=True, the same processor call, greedy
generate, only the newly written tokens decoded, memory freed
after every ask).

What changes is the battery: Step 4 runs the Series Two
diagnostic set, the same 87 asks the MiniCPM and OLA Series Two
notebooks run (Test 2 the 2x2, 48 asks; Test 3 yes/no scoring,
30 asks; Test 4 tone-shift timing, 9 asks), on the same files,
with the same question forms and the same 33-column CSV schema,
saving series2_qwen_results.csv in the clips folder. Kobby's
ruling of Nov 2, 2026 put this rerun next: Claude's council test
named the open models already on the table (both Qwens and
MiniCPM), with OLA riding along, and the first Series Two pass
ran MiniCPM and OLA only.

Two Qwen feed questions had to be settled from the pinned code
before this build, and both rulings are written into the
notebook itself, in the title cell, the Step 3 and Step 4 notes,
and every affected row's feed text:

1. The sound-alone arm of Test 2. Qwen2.5-Omni's own input form
   supports it directly. In qwen-omni-utils 0.0.8,
   process_audio_info handles an {"type": "audio", "audio":
   path} element and a video element's soundtrack in the same
   function, decoding both with the same call, librosa.load at
   SAMPLE_RATE 16000, mono. The processor
   (processing_qwen2_5_omni.py) treats audio as an independent
   optional input: text plus audio with no images and no videos
   is a supported call. So the arm is implemented through that
   documented form: the conversation carries an audio element
   naming the same swap file and no video element, the processor
   is called with text and audio only, and generate runs without
   use_audio_in_video. The decoded soundtrack is the same file's
   soundtrack through the same decoder, and the feed proof
   measures the exact waveform array (audios[0]) that the same
   process_mm_info call handed to the processor.

2. The yes score. EXPOSED. In transformers 4.52.4,
   Qwen2_5OmniForConditionalGeneration.generate
   (modeling_qwen2_5_omni.py, line 4412) routes every
   unprefixed keyword through shared_kwargs into the thinker's
   kwargs (lines 4492 and 4496). return_audio defaults to
   self.has_talker (line 4458), and the Series One load path
   calls disable_talker(), which sets has_talker to False
   (lines 4355 to 4360), so generate_audio is False (line
   4505) and the method returns self.thinker.generate(...)
   untouched (lines 4510 to 4513). The thinker is a standard
   GenerationMixin model, so asking that same greedy call for
   return_dict_in_generate and output_scores returns its result
   object with .sequences and .scores, and the first score row
   is the distribution at the first generated position. The
   flags cannot change the prompt, the feed, or the tokens
   chosen under greedy decoding; they only keep numbers the
   call already computes. So yes/no asks record the probability
   of a yes token at the first generated position, read exactly
   as the OLA Series Two runner reads its own. No internal
   path, no hooks, no extra forward pass.

One wrapper fact the runner has to respect, found in the same
source: the thinker's max_new_tokens is pre-seeded from the
wrapper's own thinker_max_new_tokens parameter (default 1024),
and the merge at line 4496 only fills keys that are missing, so
an unprefixed max_new_tokens (the form the Series One notebooks
passed) is silently ignored by this wrapper. The Series Two
job list's caps are real requirements, so the runner passes
each cap through the wrapper's documented knob,
thinker_max_new_tokens. The prompt, the feed, and the greedy
decoding are untouched by this; the Series One answers all
ended at their end token long before either cap, so nothing
about the Series One record changes.

The feed proof follows the audit rule: printed from inside the
session, per ask, before the answer is taken, and measured on
the model's own feed path, which here means the exact decoded
waveform array process_mm_info returns for that ask (duration,
sample rate 16000 by the loader's construction, sample count,
RMS in dB, nonzero share, and the same numbers per one-second
slice of that waveform; the unit columns carry those slice
numbers, and the padding column records the final slice's
shortfall against a full second, which is proof bookkeeping,
not padding the model applies, since the processor takes the
waveform whole). The frame counts the proof carries come from
the videos tensor that same call returns. A copy of each
decoded track is saved beside the results and the track is
hashed. The hard stop carries the same thresholds as the
MiniCPM and OLA Series Two runners: a sound-carrying track
that decodes missing, empty, short by more than 0.5 s against
the file's own duration, or all zeros stops the run with exit
code 7 and the rows so far saved; declared silence is exempt
by name in the job list and is itself verified to be zeros.

The sanity gate keeps the Series One Qwen form, a plain text
question through the real processor and generate path, with
the word ready required, as the task for this build specifies;
media arrival is not what that gate proves, and the notebook
says so: arrival is proved per ask by the FEED PROOF lines.

The battery's question strings are not retyped. The repaired
JSON question is pulled from pilot2/run_eggs.py at build time
and asserted at 444 characters, and the whole job list section
of the runner (sounds, prompts, targets, lures, repeats, token
caps) is copied word for word from the MiniCPM Series Two
runner, so the runs ask byte-identical questions by
construction. The build verification then proves it again by
executing both job lists and comparing them.

Plain language throughout, no em dashes.
"""
import copy
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT2 = os.path.join(HERE, "..", "pilot2")
BASE_NB = os.path.join(HERE, "project_k_qwen_omni_colab.ipynb")


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

# The Series One Qwen notebook cells are: 0 title, 1 Step 1
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
# Project K, Series Two: Qwen2.5-Omni-3B rides the diagnostic cells

**Why this notebook exists.** Series One asked four models the same
questions on the same clips, and each one lost the sound in its own
shape. Series Two asks the next question: which stage loses it? The
council's tests change one thing at a time on the same files. Feed
integrity first (did the sound provably reach the model), then
recognition, interference, reporting, and timing, as hypotheses to
separate, not labels to assume. When the council proposed the
yes/no test, it named the open models already on the table: both
Qwens and MiniCPM, with OLA riding along. The first Series Two
pass ran MiniCPM and OLA. This notebook is the correction,
ordered on Nov 2, 2026: the Qwens rerun through the same cells
next, before any new model joins the table.

**What this notebook does.** It runs Qwen2.5-Omni-3B through the
same 87 diagnostic asks MiniCPM and OLA faced:

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

Results save as `series2_qwen_results.csv` in the Drive clips
folder, in the same schema as the other Series Two results, so
the runs read side by side, row for row.

**What is reused, and what is new.** The environment and the ask
path are the Series One Qwen notebook's own, unchanged: the same
session installs with the same pins (transformers 4.52.4 among
them, now proved in Step 1), the same loading candidates (the 3B
in 16-bit form first, the 7B AWQ build second), the talker turned
off, and the same feed (one frame per second, frames capped at
100352 pixels, the video's full sound track carried in with the
picture). New in Series Two: the per-ask feed proof, the
diagnostic battery, the sound-alone arm, the yes scores, and
full load tracebacks kept in `series2_qwen_load_log.txt`.

**The doors, stated up front.** Both arms ride the model's own
processor path, the one Series One used:

- **With the picture:** the Series One form exactly. The
  conversation carries the video file and the question;
  `process_mm_info` (with `use_audio_in_video=True`) checks the
  file carries an audio stream and decodes its soundtrack,
  samples the frames, and the processor builds the model's
  inputs.
- **Sound alone:** the model's own documented audio input. The
  conversation carries an audio element naming the same file
  and no video element. The model's tools decode an audio
  element and a video soundtrack in the same function, with the
  same decode call (librosa at 16 kHz, mono), so the soundtrack
  that arrives is the same file's soundtrack through the same
  decoder; the processor is then called with text and audio
  only, and generate runs without the in-video flag. The exact
  difference is stated here and in every affected row's feed
  text: no picture reaches the model, and the prompt carries an
  audio token in place of video tokens.

**The feed proof rule.** Before every answer is taken, the runner
prints the proof for that exact feed, measured on the model's
own feed path: the exact decoded waveform array that
`process_mm_info` returns for that ask, and hands to the
processor, is measured where it sits (duration, sample rate,
sample count, RMS in dB, nonzero share, and the same numbers per
one-second slice of that waveform), and the frame count comes
from the video tensor the same call returns. The unit columns
of the CSV carry those one-second slice numbers; the padding
column records how short the final slice falls of a full
second, which is proof bookkeeping only, because the processor
takes the waveform whole and cuts nothing into seconds. A copy
of each decoded track is saved beside the results and the track
is hashed. If a track that should carry sound decodes missing,
empty, short by more than 0.5 seconds against the file's own
duration, or all zeros, the run hard-stops with a loud banner
and the rows so far are saved. Declared silence and declared
muted versions are exempt by name in the job list, never by
accident, and declared silence is itself verified to be zeros.

**The yes-score fact, settled from the code before any run.**
MiniCPM's chat door returns decoded text only, so its Series Two
rows read "not exposed". This model's door is different, and the
code says so. Its generate method is a wrapper: it sorts the
call's keywords and hands them to the thinker, a standard
Hugging Face generation model, and because the load turns the
talker off (as Series One did), the wrapper returns the
thinker's own result untouched. Asking that same greedy call to
also return its per-step scores changes nothing about the
prompt, the feed, or the tokens chosen; it only keeps numbers
the call already computes. So on yes/no asks this runner records
the probability of a yes token at the first generated position,
read from that call's own first score row, exactly the way the
OLA Series Two runner reads its own. No internal path is
substituted: no hidden states, no extra forward pass, no hooks.
Rows that are not yes/no asks carry no score, because that
question form has no yes token to score. One more wrapper fact
the runner respects: the token cap has to travel through the
wrapper's own thinker setting, because the wrapper pre-fills
that setting and ignores a plain cap passed the Series One way;
the runner uses the wrapper's knob, and the prompt, the feed,
and the greedy decoding are untouched by it.

**The sanity gate.** Step 3 ends with the Series One gate, kept
in its own form: one plain text question through the real
processor and generate path, "Reply with the word ready.", with
the word ready required in the answer. That gate proves the
model can write through this path. It does not prove sound
arrives; the per-ask feed proof is what proves that, on every
ask, before the answer is taken.

**Where it runs.** On the paid Colab account (kobbytti), like
the other Series Two runs. The 3B in 16-bit form is the size
the Series One run loaded on a T4-class chip; a bigger chip
only makes the same run faster.

**How to run it.** Run the cells from the top, in order. Step 1
sets up and proves the stack. Step 2 gets the twelve clips and
the Series Two test files. Step 3 writes the runner and runs its
check mode (proof, loading candidates, sanity). Step 4 runs the
diagnostic battery and saves `series2_qwen_results.csv`.

**The one rule.** We claim only what we ran. Every result here
is about these clips, this model, this feed, this run. Nothing
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
Series One Qwen notebooks used five and twelve of them):

1. `P02_07_5.mp4`: the vacuum clip.
2. `P32_01_54.mp4`: the jar clip.
3. `own-01.mp4`: your eggs recording.
4. `probe_speech.mp4`: the spoken sentence control.
5. `probe_tone.mp4`: three loud beeps.
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
"""))

cells.append(code(r"""
# --- Get the clips and the Series Two test files -------------------------------
# This cell only moves and, if needed, builds files. The gray and
# tone builds below are the MiniCPM Series Two notebook's Step 2
# algorithm, unchanged, used only for files the Drive folder does
# not already hold.
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
             "built_by": "qwen series2 step 2 fallback (minicpm algorithm)"}
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
                 "built_by": "qwen series2 step 2 fallback (minicpm algorithm)"}
        build_log.append(entry)
        save_both(out)
        print("tone (built):", entry)

if build_log:
    log_path = os.path.join(WORK, "series2_qwen_build_log.json")
    with open(log_path, "w") as fh:
        json.dump(build_log, fh, indent=2)
    if os.path.isdir(DRIVE_CLIPS):
        shutil.copy(log_path, os.path.join(
            DRIVE_CLIPS, "series2_qwen_build_log.json"))
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
# /content/qwen_series2_run.py and runs under the notebook
# session's own Python. That is the Qwen notebooks' own pattern:
# their stack is installed into the session (Step 1), so the
# session interpreter is the environment, and there is no private
# venv to run under, unlike the MiniCPM and OLA notebooks. The
# runner holds the proof, the Series One loading candidates, the
# sanity gate, the strict feed proof, and the Series Two
# diagnostic battery. The battery section (job list, prompts,
# targets, lures) is copied word for word from the MiniCPM
# Series Two runner.
RUNNER_SRC = r"""
# qwen_series2_run.py - the Series Two diagnostic runner for
# Qwen2.5-Omni-3B. Written by project_k_qwen_series2_colab.ipynb.
# It runs under the notebook session's own Python: the Qwen
# notebooks install their stack into the Colab session (Step 1),
# and there is no private environment to run under. Modes:
# "check" runs the proof, the loading candidates, and the sanity
# question, then stops. "battery" does all of that and then runs
# the Series Two diagnostic tests and saves the CSV.
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
# and disable_talker(), which the load below calls exactly as
# Series One did, sets has_talker to False (lines 4355 to
# 4360). So generate_audio is False (line 4505) and the wrapper
# returns self.thinker.generate(...) untouched (lines 4510 to
# 4513). The thinker is a standard GenerationMixin model, so
# return_dict_in_generate and output_scores asked of that same
# greedy call come back on its result as .sequences and
# .scores, and the first score row is the distribution at the
# first generated position. The flags cannot change the prompt,
# the feed, or the tokens greedy decoding chooses; they only
# keep numbers the call already computes. MiniCPM's chat door,
# by contrast, returns decoded text only, which is why its
# Series Two rows read "not exposed". Here the score is exposed
# by the door itself, so YES_SCORE_EXPOSED is True, and yes/no
# asks record the probability of a yes token at the first
# generated position from that call's first score row.
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
# A failed candidate prints its full traceback (never a
# cut-short reason) and the same words are appended, as events
# happen, to series2_qwen_load_log.txt, in the Drive clips
# folder first and /content second, so the file is already on
# disk even if the session dies right after a candidate does.
import traceback, datetime

LOAD_LOG_PATHS = ["/content/drive/MyDrive/project-k-clips/series2_qwen_load_log.txt",
                  "/content/series2_qwen_load_log.txt"]

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
log_load("Qwen2.5-Omni-3B Series Two load log - "
         + datetime.datetime.now().isoformat(timespec="seconds")
         + " - runner mode: " + MODE + "\n")

# --- Load the model: the Series One candidates, unchanged ----------------------
# Small one first: it is the one that fits a T4-class chip, and
# it is the model the Series One run loaded. The big AWQ build
# is tried only if the small one fails, exactly as in the
# Series One notebook. Either way the weights load in 16-bit
# form (float16) with device_map="auto", the talker is turned
# off (written answers only, and it is what makes the wrapper
# return the thinker's result directly), and the processor
# comes from the same repo. Whichever candidate loads, its
# name is printed and saved with every answer.
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

# --- Sanity check: can the model write at all? ---------------------------------
# The Series One gate, in its own form: one plain text question
# through the real processor and generate path (chat template,
# processor, generate, decode only the new tokens), with the
# word ready required in the answer. It proves the model can
# write through this path before any clip runs. It does not
# prove sound arrives; the per-ask FEED PROOF in the battery is
# what proves that. An empty answer, a missing ready, or an
# error all stop the run here, loudly. Both modes run it.
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
    print("check mode done: proof, candidates, and sanity all passed.")
    print("The diagnostic battery was not run. Run Step 4 for the tests.")
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
TRACK_DIRS = ["/content/drive/MyDrive/project-k-clips/series2_qwen_extracted_tracks",
              "/content/series2_qwen_extracted_tracks"]
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
# --- Diagnostic battery ---------------------------------------------------------
# The repaired JSON question, injected at build time from
# pilot2/run_eggs.py, character for character (444 characters).
PROMPT_JSON = __PROMPT_JSON__
assert len(PROMPT_JSON) == 444, f"prompt drifted: {len(PROMPT_JSON)} chars"

print("FEED SETTINGS, with-picture asks:", FEED_WITH_PICTURE.format(
    frames="N", tokens="M"))
print("FEED SETTINGS, sound-alone asks:", FEED_AUDIO_ALONE.format(
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
    for out_path in ("/content/drive/MyDrive/project-k-clips/series2_qwen_results.csv",
                     "/content/series2_qwen_results.csv"):
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
assert "__PROMPT_" not in RUNNER_SRC, "prompt placeholders survived"

# ---------------------------------------------------------------- step 3
cells.append(md(r"""
## Step 3 · Write the runner, then check: proof, candidates, sanity

The model never runs inside this notebook's cells directly. It
runs in a plain script, `/content/qwen_series2_run.py`, under
this session's own Python. That is the Qwen notebooks' pattern:
Step 1 installed the stack into the session, so the session
interpreter is the environment. (The MiniCPM and OLA notebooks
run under private environments instead, because their stacks
demand it; the difference is stated, not hidden.) The first
cell below writes that script. The second runs it in
**check mode**.

Check mode does four things, in order:

1. **Proof again.** The versions and files the runner imported,
   with the transformers pin checked a second time: exactly
   4.52.4, or the run stops. One new line prints the yes-score
   fact: this model's generate wrapper returns the thinker's
   own result when the talker is off, so the door exposes its
   per-step scores on request.
2. **The loading candidates.** The Series One list, unchanged:
   **the 3B in 16-bit form (float16) first**, the one that fits
   a T4-class chip and the one the Series One run loaded; the
   7B AWQ build second, only if the small one fails. The
   talker is turned off on whichever loads, and the processor
   comes from the same repo. A failed candidate prints its
   full traceback, never a cut-short reason, and appends the
   same words to `series2_qwen_load_log.txt` in the clips
   folder, so a verbatim record survives the session. If both
   fail, the runner's closing lines name that file. Whichever
   candidate loads is printed and saved with every answer.
3. **The token-cap note.** The runner passes each ask's token
   cap through the generate wrapper's own thinker setting. The
   wrapper pre-fills that setting itself (at 1024) and ignores
   a plain cap, which is the form the Series One notebooks
   passed; their answers all ended at the end token long
   before either number, so their record is untouched. Here
   the job list's caps (300, 128, 32, 16) are requirements of
   the test design, so they travel by the knob the wrapper
   actually reads. Nothing else about the call changes.
4. **The sanity question.** The Series One gate in its own
   form: one plain text question through the real processor
   and generate path, "Reply with the word ready.", and the
   word ready is required in the answer. It proves the model
   can write through this path. If the answer comes back
   empty, without the word, or with an error, the script
   stops and Step 4 must not be run. Sound arrival is not
   this gate's job; Step 4's per-ask feed proof carries that.
"""))

cells.append(code(
    "# --- Write the Series Two runner script ----------------------------------------\n"
    "# The runner holds the proof, the loading candidates, the\n"
    "# sanity gate, the strict feed proof, and the diagnostic\n"
    "# battery. It is written to a plain file here and only ever\n"
    "# run under this session's own Python, the interpreter\n"
    "# Step 1 installed the stack into.\n"
    "RUNNER_SOURCE = " + repr(RUNNER_SRC) + "\n"
    "with open(\"/content/qwen_series2_run.py\", \"w\") as fh:\n"
    "    fh.write(RUNNER_SOURCE)\n"
    "print(\"runner written:\", \"/content/qwen_series2_run.py\", "
    "len(RUNNER_SOURCE), \"chars\")\n"))

cells.append(code(r"""
# --- Check mode: proof, candidates, sanity ---------------------------------------
# Runs the runner under this session's own Python and streams
# its output here live. A nonzero exit code means the proof, a
# loading candidate, or the sanity gate failed, and the run
# stops in this cell. The first check run downloads the model
# files once; that takes a while and is normal.
import subprocess, sys
proc = subprocess.run([sys.executable, "/content/qwen_series2_run.py", "check"])
print()
print("check exit code:", proc.returncode)
if proc.returncode != 0:
    print("!" * 72)
    print("STOP. THE CHECK RUN FAILED. The runner's own output above says")
    print("where: a proof line, a loading candidate, or the sanity")
    print("question. Do not run Step 4. Show this output to Muse.")
    print("!" * 72)
    raise SystemExit("STOP: runner check failed")
print("check passed: proof lines right, model loaded, sanity answered.")
print("Go on to Step 4.")
"""))

# ---------------------------------------------------------------- step 4
cells.append(md(r"""
## Step 4 · Run the diagnostic battery

The next cell runs the runner in **battery mode**: proof and
candidates again, sanity again, then the three Series Two
tests, the same 87 asks, files, and question forms as the
MiniCPM and OLA Series Two runs. Every ask prints its FEED
PROOF line before the model's answer is taken. Watch for it:
that line is the whole point of Series Two.

**How this model is asked.** With-picture asks ride the Series
One path, unchanged: the conversation carries the video file
at one frame per second with frames capped at 100352 pixels,
`process_mm_info` carries the video's own sound track in, and
the processor builds the inputs. Sound-alone asks (the first
arm of Test 2) ride the model's own documented audio input:
an audio element naming the same file, decoded by the same
function at the same 16 kHz mono, with the processor called
on text and audio only. Decoding is greedy everywhere. Token
caps per ask are the job list's own: 300 for the open
question, 32 for presence, 128 for detail, 16 for yes/no, 32
for the timing asks, passed through the wrapper's thinker
setting as Step 3 explains.

**Test 2, the 2x2 (48 asks).** Sounds in a fixed order: speech
first (the baseline), then true silence (the negative
control), then water, then beeps. Each sound is fed two ways:
sound alone, and the same soundtrack with the swap still
picture. Each arm is asked three ways, two repeats each: the
open repaired JSON question (word for word the Series One
question), a direct presence question made for that sound,
and a direct detail question (for speech: "Write down any
words spoken in this clip."; for the others, a short
sound-name question, and for silence an any-sound control).

**Test 3, yes/no scoring (30 asks).** All seven swap
soundtracks, three versions each: sound on with the still
picture, muted with the still picture, and sound on with a
plain gray frame. Each version is asked about its matching
sound ("Is there the sound of X in this clip? Answer yes or
no."), the mixed file about both of its sounds, and two lures
ride along: the speech file is asked about water, and the
water file is asked about speech. The reading is fixed in
advance in the Series Two plan: yes not moving between sound
on and muted means the sound never got in; yes rising while
the free answer still says none means the loss is at
reporting; yes staying high on a muted kitchen picture means
the scene is guessing. Like OLA and unlike MiniCPM, the
yes/no rows here also carry a measured score: the probability
of a yes token at the first generated position, read from the
same greedy generate call's own returned scores. The words
stay the primary record; the score is the lean behind the
word, and it exists only because this model's door exposes
it.

**Test 4, tone-shift timing (9 asks).** The three tone files
from Step 2. Per file, three separate asks: presence (yes or
no), early-or-late ("Does the tone start in the first half or
the second half of the clip?"), and numeric ("At what second
does the tone start? Answer with a number."). Timing gets
scored later, only on files where presence was detected, so a
timing failure can no longer hide inside a hearing failure.

**Hard stops and warnings.** A feed that fails its proof
stops the whole run on the spot (exit code 7), with the rows
completed so far saved first. An ask that errors or comes
back empty is saved with its full error text in `raw_answer`
and a WARNING on screen, never as silence, and the run ends
with the RUN PROBLEM banner (exit code 6). Results save to
`series2_qwen_results.csv` in the Drive clips folder, one row
per ask, with the full feed-proof numbers on every row, in
the same schema as the other Series Two results.
"""))

cells.append(code(r"""
# --- Battery mode: the Series Two diagnostic tests --------------------------------
# Same runner, same session Python, streamed live. The runner
# saves the CSV itself, to the Drive clips folder and to
# /content. A nonzero exit code means the pre-flight gates
# failed up front, a feed proof hard-stopped the run, or the
# run finished with broken asks (the banners above say which).
import subprocess, sys
proc = subprocess.run([sys.executable, "/content/qwen_series2_run.py", "battery"])
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

Read this run against the MiniCPM and OLA Series Two runs, row
for row, and against this model's own Series One answers. The
files were identical and the questions were byte-identical, so
a difference is a model difference or a feed difference, and
the feed differences are written down, not hidden. They are:

- **The doors.** MiniCPM was fed one frame per second with the
  matching second of sound, through its unified chat. OLA was
  fed up to 64 frames per clip and the whole clip's sound
  through two encoders at once. This model is fed one frame
  per second with frames capped small and the full sound
  track, the Series One Qwen feed: a thinner picture than
  OLA's, the same whole soundtrack. A miss here could still
  be a small-picture miss; a hit cannot be explained away by
  the picture feed, because the sound track was never
  reduced.
- **The sound-alone arms.** Each model's arm rides its own
  documented door: MiniCPM's unified chat with no picture,
  OLA's audio+text branch, and here the processor's audio
  element form, decoded by the same function and the same
  call as the in-video soundtrack. The arms are comparable
  in what reaches the model (the soundtrack, no picture) and
  different in machinery, and the rows say which machinery.
- **The yes scores.** MiniCPM's chat door exposes no score,
  so its yes/no rows are words only. OLA's door and this
  door both expose the score, so their yes/no rows carry the
  lean behind the word. Compare words to words across all
  three runs; compare scores only where they exist.
- **The unit columns.** In MiniCPM's CSV the unit columns
  count one-second units of its feed. In OLA's they count
  30 second chunks. Here they count one-second slices of the
  exact decoded waveform the processor received, measured as
  proof; the column names match by design, and each
  notebook says what its unit is.

Then the tests themselves:

- **Feed proof first.** A row is only as good as its proof
  line. Rows after a hard stop do not exist, on purpose: a
  stopped run is a named failure, not a pile of quiet none
  answers.
- **Test 2 reads as a square.** Sound alone against sound
  with a picture separates the picture's interference; open
  against direct separates the question's suppression.
  Speech is the baseline and silence is the control: a yes
  on silence is a false alarm, and the rate matters as much
  as the hits. MiniCPM's square said the loss sits in the
  telling, not the hearing; OLA's square showed invention
  on silence instead. This model's square either agrees
  with one shape or shows a third.
- **Test 3 reads by movement.** Compare each sound across
  its three versions before comparing sounds to each other,
  and read the yes score beside the word: a word that flips
  while the score barely moves, or a score that moves while
  the word does not, is itself a finding about the
  reporting stage.
- **Test 4 reads in order.** Presence first. Early-or-late
  and numeric starts count only where presence was a hit.
  MiniCPM placed the tones +2, -1, +2; OLA placed them -1,
  -5, -7; three tones is a small cell, stated as such, for
  every model.
- **Scope.** Every claim stays capped at this model, this
  feed, these files, this date. The stages in the Series
  Two plan stay hypotheses until an intervention separates
  them, and these three tests are the interventions. The
  environment is part of the record too: the Colab session
  install with transformers pinned at 4.52.4, no private
  environment, and the environment string is saved in the
  `model` column of every CSV row.

When you finish, the file `series2_qwen_results.csv` in your
Drive is the record. Send it back, and the Series Two runs
get written up side by side.
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

out = os.path.join(HERE, "project_k_qwen_series2_colab.ipynb")
with open(out, "w") as fh:
    json.dump(nb, fh, indent=1)
print("wrote", out, "cells:", len(cells))
