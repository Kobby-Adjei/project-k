# build_indexcheck_qwen_notebook.py
# Builds project_k_qwen_indexcheck_colab.ipynb: the cheap checks
# the council asked for before anything else (Nov 4, 2026,
# Claude's order, Kobby's tap):
#   Part A: the layer-35 index check. All-positions patches at
#     layers 0, 18, 28, 35 must reproduce the donor's yes score
#     exactly (after a full-row replacement nothing of the
#     recipient remains, so any later layer sees only donor
#     states). Final-position-only cells at 35 (as built, and
#     with the donor state index shifted by one), 34, and 28
#     localize the anomaly the audit found. A norm diagnostic
#     settles whether hidden_states[36] is pre- or post-norm in
#     this stack by reproducing the forward's own logits two
#     ways.
#   Part B: the model is finally ASKED about the properties.
#     The 360 property tones (same files the step 5 probes read)
#     get two yes/no questions, pitch and onset, scored from the
#     first-token logits through the same feed the probes used.
#     Plus the missing baseline: all 8 unpatched patching tones
#     under the open question.
# The runner head (proof, load, scorer, feed, token check, feed
# proof, baselines, captures, patched_generate) is the Handover
# Correction rerun's own machinery, taken whole so the check
# tests the suspect code as built; only three surgical changes:
# a state_index override and an "all" mode in patched_generate,
# a run-log tee, and indexcheck_ file names.
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))

_ns = {"__file__": os.path.join(HERE, "build_handover_rerun_notebook.py"),
       "__name__": "rerun_builder"}
exec(open(os.path.join(HERE, "build_handover_rerun_notebook.py")).read(), _ns)
RERUN_SRC = _ns["RUNNER_SRC"]
RERUN_CELLS = _ns["cells"]

SWEEP_MARKER = "# --- The sweep, checkpointed"
head = RERUN_SRC[:RERUN_SRC.index(SWEEP_MARKER)]

# --- surgery 1: state_index override in patched_generate --------
old_sig = ("def patched_generate(recipient, prompt, donor_cap, layer, mode,\n"
           "                     max_tokens, want_yes):")
new_sig = ("def patched_generate(recipient, prompt, donor_cap, layer, mode,\n"
           "                     max_tokens, want_yes, state_index=None):")
assert old_sig in head
head = head.replace(old_sig, new_sig)
old_vec = '    donor_vec = donor_cap["states"][layer + 1][donor_pos]'
new_vec = ('    _si = layer + 1 if state_index is None else state_index\n'
           '    donor_vec = donor_cap["states"][_si][donor_pos]')
assert old_vec in head
head = head.replace(old_vec, new_vec)

# --- surgery 2: the "all" patch mode -----------------------------
old_else = '    else:\n        mechanics_fail(f"unknown patch mode {mode!r}")'
new_else = ('    elif mode == "all":\n'
            '        recip_pos = list(range(len(row)))\n'
            '        donor_pos = list(range(len(donor_cap["row"])))\n'
            '    else:\n'
            '        mechanics_fail(f"unknown patch mode {mode!r}")')
assert old_else in head
head = head.replace(old_else, new_else)

# --- surgery 3: run-log tee + file names --------------------------
TEE = r'''
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
                   "indexcheck_run_log.txt", "w", buffering=1)
except Exception:
    _log_fh = open("/content/indexcheck_run_log.txt", "w",
                   buffering=1)
sys.stdout = _Tee(sys.stdout, _log_fh)

'''
anchor = "# --- The token check"
assert anchor in head
head = head.replace(anchor, TEE + anchor, 1)
head = head.replace("handover_rerun_", "indexcheck_")
head = head.replace("qwen_handover_run.py", "qwen_indexcheck_run.py")

RUNNER_SRC = head

# --- Part A --------------------------------------------------------
RUNNER_SRC += r'''
# --- Part A: the index check ---------------------------------------------------
# The audit's sanity argument (Claude, Nov 4, 2026): pasting the
# donor's state at the final position after the last decoder
# layer must reproduce the donor's answer exactly, because only
# the final norm and the output head run after it. The rerun's
# final-at-35 cells did not (about 0.52 against donor baselines
# near 0.88). This part discriminates the causes on the rerun's
# own machinery:
#   * all-positions patches at layers 0, 18, 28, 35: after a
#     full-row replacement the recipient contributes nothing, so
#     every later layer sees only donor states and the score
#     must equal the donor baseline at EVERY layer. A failure
#     pattern by layer localizes the break.
#   * final-position-only cells: layer 35 as built (donor state
#     index 36), layer 35 with the donor state index shifted to
#     35, layer 34 as built, layer 28 as built (the rerun's
#     near-donor cell, as a replication).
#   * a norm diagnostic: the forward's own last-position logits
#     are reproduced from hidden_states[36] directly and after
#     the thinker's final norm; whichever matches settles
#     whether index 36 is pre- or post-norm in this stack.
# Verdicts print mechanically; the read happens outside.
print("=" * 70)
print("PART A: THE INDEX CHECK")
print("=" * 70)

# norm diagnostic on tone_00
try:
    _inp = build_inputs("tone_00", PROMPT_BEEPS)
    _out = thinker_forward(_inp)
    _hs36 = _out.hidden_states[36][0, -1].float()
    _logits = _out.logits[0, -1].float()
    _head = THINKER.lm_head
    _d_raw = float((_head(_hs36) - _logits).abs().max().item())
    _normed = THINKER.model.norm(_out.hidden_states[36])[0, -1].float()
    _d_norm = float((_head(_normed) - _logits).abs().max().item())
    print(f"NORM DIAGNOSTIC: max |logits - head(state36)| = "
          f"{_d_raw:.6f}; max |logits - head(norm(state36))| = "
          f"{_d_norm:.6f}. The near-zero one is the pipeline "
          f"the forward itself used, so it settles whether "
          f"state 36 is pre- or post-norm here.")
    del _inp, _out
except Exception as e:
    print(f"NORM DIAGNOSTIC FAILED TO RUN: {str(e)[:200]}")
    print("(the patch cells below still discriminate the main "
          "question; this print is the record that the "
          "diagnostic did not run)")
free_memory()

INDEX_FIELDS = ["cell", "pair", "donor", "recipient", "layer",
                "mode", "state_index", "yes_score", "answer",
                "donor_baseline", "abs_diff_vs_donor"]
index_rows = []

def index_cell(cell, donor, recip, layer, mode, state_index=None):
    cap = CAPTURES[(donor, PROMPT_BEEPS)]
    dec, yp = patched_generate(
        recip, PROMPT_BEEPS, cap, layer, mode,
        PRESENCE_MAX_TOKENS, True, state_index=state_index)
    db = baseline_yes[donor]
    diff = abs(yp - db)
    index_rows.append({
        "cell": cell, "pair": f"{donor}>{recip}", "donor": donor,
        "recipient": recip, "layer": layer, "mode": mode,
        "state_index": (layer + 1 if state_index is None
                        else state_index),
        "yes_score": f"{yp:.4f}", "answer": dec,
        "donor_baseline": f"{db:.4f}",
        "abs_diff_vs_donor": f"{diff:.4f}"})
    print(f"  {cell}: {donor}>{recip} layer {layer} mode {mode} "
          f"state_index {(layer + 1 if state_index is None else state_index)}: "
          f"yes {yp:.4f} (donor baseline {db:.4f}, diff "
          f"{diff:.4f}), answer {dec!r}")
    free_memory()

PAIRS4 = [(f"tone_{i:02d}", f"silence_{i:02d}") for i in range(4)]
for layer in (0, 18, 28, 35):
    for donor, recip in PAIRS4:
        index_cell("all_positions", donor, recip, layer, "all")
for donor, recip in PAIRS4:
    index_cell("final_as_built_35", donor, recip, 35, "final")
for donor, recip in PAIRS4:
    index_cell("final_shifted_35", donor, recip, 35, "final",
               state_index=35)
for donor, recip in PAIRS4:
    index_cell("final_as_built_34", donor, recip, 34, "final")
for donor, recip in PAIRS4:
    index_cell("final_as_built_28", donor, recip, 28, "final")

def write_index(p):
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=INDEX_FIELDS)
        w.writeheader()
        w.writerows(index_rows)
save_both("indexcheck_results.csv", write_index)
print("PART A READOUT (mechanical):")
for layer in (0, 18, 28, 35):
    diffs = [float(r["abs_diff_vs_donor"]) for r in index_rows
             if r["cell"] == "all_positions"
             and int(r["layer"]) == layer]
    print(f"  all-positions layer {layer}: max diff vs donor "
          f"{max(diffs):.4f} over {len(diffs)} pairs "
          f"(exact = 0.0000 up to float noise)")
for cell in ("final_as_built_35", "final_shifted_35",
             "final_as_built_34", "final_as_built_28"):
    diffs = [float(r["abs_diff_vs_donor"]) for r in index_rows
             if r["cell"] == cell]
    print(f"  {cell}: max diff vs donor {max(diffs):.4f}, "
          f"mean {sum(diffs) / len(diffs):.4f}")

'''

# --- Part B --------------------------------------------------------
RUNNER_SRC += r'''
# --- Part B: ask the model about the properties --------------------------------
# Step 5 showed pitch and onset stay readable inside the model.
# Nobody had asked the model itself. Here the 360 property
# tones (the same files the probes read, regenerated in the
# property battery and copied from Drive in Step 2) get two
# yes/no questions through the same feed form, scored from the
# first-token logits with the same yes scorer, plus the
# normalized yes share over the Yes/No tokens as the cleaner
# readout the audit asked for. Truth comes from the property
# manifest (pitch_band, onset_band). Also here: the missing
# baseline, all 8 unpatched patching tones under the open
# question, to stand beside patching_openform.csv.
print("=" * 70)
print("PART B: BEHAVIORAL PROPERTY QUESTIONS")
print("=" * 70)

NO_TOKEN_IDS = []
for _word in ("No", " No"):
    _enc = processor.tokenizer.encode(_word,
                                      add_special_tokens=False)
    if _enc:
        NO_TOKEN_IDS.append(int(_enc[0]))
NO_TOKEN_IDS = sorted(set(NO_TOKEN_IDS))
print("no token ids for scoring:", NO_TOKEN_IDS,
      [processor.tokenizer.decode([i]) for i in NO_TOKEN_IDS])

PITCH_Q = ("Is the tone in this clip high-pitched? "
           "Answer yes or no.")
ONSET_Q = ("Does the tone start in the first second of this "
           "clip? Answer yes or no.")

def forward_score(path, prompt):
    inputs, _ = feed_inputs(path, prompt, expect_silence=False)
    out = thinker_forward(inputs)
    logits = out.logits[0, -1].float()
    probs = torch.softmax(logits, dim=-1)
    p_yes = sum(float(probs[i].item()) for i in YES_TOKEN_IDS)
    p_no = sum(float(probs[i].item()) for i in NO_TOKEN_IDS)
    first = processor.tokenizer.decode(
        [int(torch.argmax(logits).item())])
    share = p_yes / (p_yes + p_no) if (p_yes + p_no) > 0 else 0.0
    del out, inputs
    return p_yes, share, first

BEHAV_FIELDS = ["clip", "pitch_band", "onset_band", "question",
                "truth_yes", "yes_score", "yes_share",
                "first_token", "correct_by_score",
                "correct_by_share", "correct_by_word"]
behav_rows = []
with open("/content/property_manifest.csv", newline="") as fh:
    prop_rows = [r for r in csv.DictReader(fh)
                 if r["group"] == "tone"]
print(f"property tones to ask: {len(prop_rows)}")
for i, r in enumerate(prop_rows):
    path = os.path.join("/content/property_clips",
                        r["clip_name"] + ".wav")
    for qkey, question, truth in (
            ("pitch", PITCH_Q, r["pitch_band"] == "high"),
            ("onset", ONSET_Q, r["onset_band"] == "early")):
        try:
            p_yes, share, first = forward_score(path, question)
        except FeedError as e:
            print()
            print("!" * 70)
            print("PART B FEED HARD STOP:")
            print(str(e))
            print(f"Clip {r['clip_name']} was not asked, and "
                  f"the run stops here. No padding was "
                  f"substituted. Show this whole output to "
                  f"Muse.")
            print("!" * 70)
            sys.exit(7)
        word_yes = first.strip().lower().startswith("yes")
        behav_rows.append({
            "clip": r["clip_name"],
            "pitch_band": r["pitch_band"],
            "onset_band": r["onset_band"], "question": qkey,
            "truth_yes": truth,
            "yes_score": f"{p_yes:.4f}",
            "yes_share": f"{share:.4f}", "first_token": first,
            "correct_by_score": (p_yes >= 0.5) == truth,
            "correct_by_share": (share >= 0.5) == truth,
            "correct_by_word": word_yes == truth})
    if (i + 1) % 60 == 0:
        print(f"  asked {i + 1}/{len(prop_rows)} tones "
              f"(both questions each)")
    free_memory()

def write_behav(p):
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=BEHAV_FIELDS)
        w.writeheader()
        w.writerows(behav_rows)
save_both("behavior_property_results.csv", write_behav)
print("PART B READOUT (mechanical):")
for qkey in ("pitch", "onset"):
    rs = [r for r in behav_rows if r["question"] == qkey]
    for key in ("correct_by_score", "correct_by_share",
                "correct_by_word"):
        acc = sum(1 for r in rs if r[key] in (True, "True")) / len(rs)
        print(f"  {qkey} [{key}]: {acc:.3f} over {len(rs)} asks")

print("OPEN FORM, ALL 8 UNPATCHED PATCHING TONES:")
open_rows = []
for i in range(8):
    clip = f"tone_{i:02d}"
    dec, _ = ask_unpatched(clip, PROMPT_OPEN, OPEN_MAX_TOKENS,
                           False)
    open_rows.append({"clip": clip, "answer": dec})
    print(f"  {clip}: {dec!r}")

def write_open(p):
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["clip", "answer"])
        w.writeheader()
        w.writerows(open_rows)
save_both("openform_tones_unpatched.csv", write_open)

print("=" * 70)
print("battery done: index check cells run, property questions")
print("asked on all 360 tones, open form taken on all 8 tones.")
print("Outputs: indexcheck_results.csv, "
      "behavior_property_results.csv, "
      "openform_tones_unpatched.csv, indexcheck_run_log.txt, "
      "plus the head's baselines and token check files.")
'''

compile(RUNNER_SRC, "qwen_indexcheck_run.py", "exec")
print("runner compiles, chars:", len(RUNNER_SRC))

# ---------------------------------------------------------------- cells
def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.splitlines(keepends=True)}

def code(text):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}

cells = []
cells.append(md(r"""# Project K, Series Three: the cheap checks, Qwen2.5-Omni-3B

**What this is.** The council's audit of the corrected Series
Three results (Nov 4, 2026) found one suspect cell and one
untested claim. Before any new model or any bigger test, two
cheap checks, in one battery:

- **Part A, the index check.** Pasting a donor's inside numbers
  into a silent run at the last layer, at the answer's own slot,
  should copy the donor's answer exactly; the rerun's cells said
  about 0.52 against a donor baseline near 0.88. Part A pastes
  at **all** positions at layers 0, 18, 28, 35 (after a full-row
  paste nothing of the silent run remains, so the score must
  equal the donor's at every layer), repeats the final-slot
  cells as built and with the donor index shifted by one, and
  runs a norm diagnostic that settles what the last state index
  holds in this stack.
- **Part B, ask the model.** The step 5 checkers read pitch and
  start time perfectly inside the model, but the model itself
  was never asked. All 360 property tones get two yes/no
  questions (high-pitched? starts in the first second?), scored
  two ways, and all 8 patching tones finally get the open
  question unpatched, the missing baseline.

**The machinery is the rerun's own** (proof, load, scorer, feed,
token check, baselines, captures, patch code), taken whole so
the check tests the suspect code as built. Two additions only:
an "all positions" patch mode and a donor-index override, both
used only by Part A's cells.

**Where it runs.** Free Colab account (kadjei1), per the
routing rule. **How.** Run the cells from the top, in order.

**The one rule.** We claim only what we ran.
"""))
cells.append(RERUN_CELLS[1])   # Step 1 markdown
cells.append(RERUN_CELLS[2])   # Step 1 code (installs)
cells.append(md(r"""## Step 2 · Get the clips

Two sets, both from the Drive folder `tracing-qwen3b`:

- the **patching clips** (`patching_clips`: 20 WAVs and their
  manifest), the surgery battery's own files;
- the **property clips** (`property_clips`: 420 WAVs) and
  `property_manifest.csv`, the step 5 battery's files. Part B
  asks only the 360 tones; the manifest carries each tone's
  true pitch band and onset band.

The next cell brings the patching clips; the one after it
brings the property files.
"""))
cells.append(RERUN_CELLS[4])   # brings patching clips
cells.append(code(r"""# --- Get the property clips and manifest -----------------------------------
import os, shutil

WORK = "/content/property_clips"
DRIVE_CLIPS = "/content/drive/MyDrive/tracing-qwen3b/property_clips"
os.makedirs(WORK, exist_ok=True)

try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception as e:
    print("Drive mount skipped:", e)

names = sorted(n for n in os.listdir(DRIVE_CLIPS)
               if n.endswith(".wav"))
print("property clips on Drive:", len(names))
copied = 0
for name in names:
    dest = os.path.join(WORK, name)
    if not os.path.exists(dest):
        shutil.copy(os.path.join(DRIVE_CLIPS, name), dest)
        copied += 1
print("copied:", copied, "| in place:", len(os.listdir(WORK)))

src = "/content/drive/MyDrive/tracing-qwen3b/property_manifest.csv"
shutil.copy(src, "/content/property_manifest.csv")
print("manifest copied:", os.path.getsize(
    "/content/property_manifest.csv"), "bytes")
"""))
cells.append(md(r"""## Step 3 · Write the runner, then check

The runner is written to a plain file and run under this
session's own Python. Check mode runs the proof, the loading
candidates, the layer resolution, the token check, and the
feed proof over the patching clips, then stops. Nothing in
Part A or Part B runs in check mode.
"""))
_write = ''.join(RERUN_CELLS[6]["source"]).replace(
    "qwen_handover_run.py", "qwen_indexcheck_run.py").replace(
    "the patching runner script", "the index-check runner script").replace(
    "the baselines, the donor\n# captures, the patching sweep, and the\n# open-form check",
    "the baselines, the donor\n# captures, the index-check cells, and the\n# behavioral property questions")
# swap in this build's runner source
import re as _re
_write = _re.sub(r"RUNNER_SOURCE = '(?:[^'\\]|\\.)*'",
                 lambda m: "RUNNER_SOURCE = " + repr(RUNNER_SRC),
                 _write, count=1)
cells.append(code(_write))
_check = ''.join(RERUN_CELLS[7]["source"]).replace(
    "qwen_handover_run.py", "qwen_indexcheck_run.py")
cells.append(code(_check))
cells.append(md(r"""## Step 4 · Run the battery

Battery mode runs the baselines and captures (the rerun's
head), then Part A (the index check) and Part B (the property
questions and the open-form tones). Everything prints live and
is also written to `indexcheck_run_log.txt` in the Drive
folder. Results land in the same folder:
`indexcheck_results.csv`, `behavior_property_results.csv`,
`openform_tones_unpatched.csv`.
"""))
_batt = ''.join(RERUN_CELLS[9]["source"]).replace(
    "qwen_handover_run.py", "qwen_indexcheck_run.py").replace(
    "the surgery battery", "the index check and the property questions")
cells.append(code(_batt))
cells.append(md(r"""## Closing · What to send back

From the Drive folder `tracing-qwen3b`, the run's own words:

1. the NORM DIAGNOSTIC line;
2. the PART A READOUT block (max diff per cell);
3. the PART B READOUT block (accuracy per question per
   readout) and the 8 open-form answers;
4. the battery exit code;
5. whether the three results CSVs exist in the folder.
"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3",
                                  "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(HERE, "project_k_qwen_indexcheck_colab.ipynb")
with open(out, "w") as fh:
    json.dump(nb, fh)
print("wrote", out, "cells:", len(cells))
