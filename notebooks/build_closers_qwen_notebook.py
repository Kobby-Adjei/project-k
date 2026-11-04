# build_closers_qwen_notebook.py
# Builds project_k_qwen_closers_colab.ipynb: the two closers
# the council set before the property patch (Nov 4, 2026;
# Kobby's go).
#   Part 1, the one-cell top check (Claude): paste the donor's
#     entry-36 state (the post-norm final state) into the
#     recipient's post-norm slot at the final position. Only
#     the output head follows, so the yes score must return the
#     donor baseline exactly. Closes the index account
#     empirically and validates the state-36 probe readout.
#   Part 2, the wording battery (ChatGPT): the behavioral
#     result rests on one wording per question. The 360
#     property tones get the reversed yes/no questions and
#     direct forced choices (high/low, early/late). Reporting
#     follows the corrected standard: accuracy by word, by the
#     normalized share at 0.5, and the threshold-free AUC.
# The runner head is the index-check runner's own head (proof,
# load, scorer, feed, token check, baselines, captures), taken
# whole; only the file names change (indexcheck_ to closers_).
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))

_ns = {"__file__": os.path.join(HERE, "build_indexcheck_qwen_notebook.py"),
       "__name__": "indexcheck_builder"}
exec(open(os.path.join(HERE, "build_indexcheck_qwen_notebook.py")).read(), _ns)
IDX_SRC = _ns["RUNNER_SRC"]
IDX_CELLS = _ns["cells"]

PART_A_AT = IDX_SRC.index("# --- Part A: the index check")
head = IDX_SRC[:PART_A_AT]
head = head.replace("indexcheck_", "closers_")
head = head.replace("qwen_indexcheck_run.py", "qwen_closers_run.py")

RUNNER_SRC = head

RUNNER_SRC += r'''
# --- Part 1: the one-cell top check --------------------------------------------
# Claude's closing cell for the index account (Nov 4, 2026):
# paste the donor's entry-36 state (the post-norm final state)
# into the recipient's post-norm slot at the final position.
# Only the output head runs after that slot, so the yes score
# must return the donor's generate baseline exactly:
# 0.8811 / 0.8870 / 0.7986 / 0.8781 for tone_00..03. Computed
# by forward only: the recipient's own hidden-state tuple is
# cloned, its final-position entry 36 is replaced with the
# donor's, and the head reads the result. Self lines (a clip's
# own state through the same clone construction, and the raw
# forward logits) must also reproduce the baselines; they are
# the construction's own sanity.
print("=" * 70)
print("PART 1: THE POST-NORM TOP CHECK")
print("=" * 70)

def yes_from_logits(logits_row):
    probs = torch.softmax(logits_row.float(), dim=-1)
    return sum(float(probs[i].item()) for i in YES_TOKEN_IDS)

TOP_FIELDS = ["direction", "donor", "recipient", "score",
              "donor_baseline", "abs_diff_vs_donor"]
top_rows = []

def top_cell(direction, donor, recip):
    d_in = build_inputs(donor, PROMPT_BEEPS)
    d_out = thinker_forward(d_in)
    r_in = build_inputs(recip, PROMPT_BEEPS)
    r_out = thinker_forward(r_in)
    hs = r_out.hidden_states[36].clone()
    hs[0, -1] = d_out.hidden_states[36][0, -1]
    swapped = yes_from_logits(THINKER.lm_head(hs[0, -1]))
    db = baseline_yes[donor]
    top_rows.append({
        "direction": direction, "donor": donor,
        "recipient": recip, "score": f"{swapped:.4f}",
        "donor_baseline": f"{db:.4f}",
        "abs_diff_vs_donor": f"{abs(swapped - db):.4f}"})
    print(f"  {direction}: {donor} into {recip}: yes "
          f"{swapped:.4f} (donor baseline {db:.4f}, diff "
          f"{abs(swapped - db):.4f})")
    del d_in, d_out, r_in, r_out, hs
    free_memory()

for i in range(4):
    # self lines first: the construction on a clip's own state
    top_cell("self", f"tone_{i:02d}", f"tone_{i:02d}")
for i in range(4):
    top_cell("tone_into_silence", f"tone_{i:02d}",
             f"silence_{i:02d}")
for i in range(4):
    top_cell("silence_into_tone", f"silence_{i:02d}",
             f"tone_{i:02d}")

def write_top(p):
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=TOP_FIELDS)
        w.writeheader()
        w.writerows(top_rows)
save_both("topcheck_results.csv", write_top)
print("PART 1 READOUT (mechanical):")
for direction in ("self", "tone_into_silence",
                  "silence_into_tone"):
    diffs = [float(r["abs_diff_vs_donor"]) for r in top_rows
             if r["direction"] == direction]
    print(f"  {direction}: max diff vs the pasted clip's "
          f"baseline {max(diffs):.4f} over {len(diffs)} cells")

'''

RUNNER_SRC += r'''
# --- Part 2: the wording battery -------------------------------------------------
# The behavioral result (Part B of the index check) rests on
# one wording per question, and the corrected account says the
# yes/no cut is misplaced. Whether the bias belongs to the
# question form or to the model is testable: ask the same
# facts other ways. The 360 property tones (the step 5 files)
# get four more forms through the same feed, scored from the
# first-token logits:
#   pitch_reversed: "Is the tone in this clip low-pitched?
#     Answer yes or no." (truth: pitch is low)
#   onset_reversed: "Does the tone start after the first
#     second of this clip? Answer yes or no." (truth: onset
#     is late; the early bursts start at 0.4-0.6 s, the late
#     at 2.4-2.6 s, so the first-second line is clean)
#   pitch_choice: "Is the tone high-pitched or low-pitched?
#     Answer with one word: high or low." (scored on the
#     high share of the high/low token pair)
#   onset_choice: "Does the tone start early or late in this
#     clip? Answer with one word: early or late." (early share)
# Reporting, per the corrected standard: accuracy by the
# emitted word, accuracy by the normalized share at 0.5, and
# the AUC of the share (threshold-free). Predictions,
# registered: if the bias lives in the yes/no form, the
# reversed questions mirror it (the model leans yes there too,
# flipping which class its word favors) while the forced
# choices, which read the margin directly, score high with
# AUC near the yes/no margin's. If the forced choices also sit
# at chance, the threshold account needs revision.
print("=" * 70)
print("PART 2: THE WORDING BATTERY")
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

def ids_for(words):
    out = []
    for w in words:
        for form in (w, " " + w, w.capitalize(),
                     " " + w.capitalize()):
            enc = processor.tokenizer.encode(
                form, add_special_tokens=False)
            if enc:
                out.append(int(enc[0]))
    return sorted(set(out))

HIGH_IDS = ids_for(["high"])
LOW_IDS = ids_for(["low"])
EARLY_IDS = ids_for(["early"])
LATE_IDS = ids_for(["late"])
print("choice token ids: high", HIGH_IDS, "low", LOW_IDS,
      "early", EARLY_IDS, "late", LATE_IDS)

PITCH_REV_Q = ("Is the tone in this clip low-pitched? "
               "Answer yes or no.")
ONSET_REV_Q = ("Does the tone start after the first second "
               "of this clip? Answer yes or no.")
PITCH_CHOICE_Q = ("Is the tone high-pitched or low-pitched? "
                  "Answer with one word: high or low.")
ONSET_CHOICE_Q = ("Does the tone start early or late in this "
                  "clip? Answer with one word: early or late.")

def forward_logits(path, prompt):
    inputs, _ = feed_inputs(path, prompt, expect_silence=False)
    out = thinker_forward(inputs)
    logits = out.logits[0, -1].float()
    first = processor.tokenizer.decode(
        [int(torch.argmax(logits).item())])
    del out, inputs
    return logits, first

def share_of(logits, ids_a, ids_b):
    probs = torch.softmax(logits, dim=-1)
    pa = sum(float(probs[i].item()) for i in ids_a)
    pb = sum(float(probs[i].item()) for i in ids_b)
    return pa / (pa + pb) if (pa + pb) > 0 else 0.0

def auc_of(pairs):
    pos = sorted(s for s, t in pairs if t)
    neg = sorted(s for s, t in pairs if not t)
    if not pos or not neg:
        return 0.0
    wins = 0.0
    import bisect
    for s in pos:
        lo = bisect.bisect_left(neg, s)
        hi = bisect.bisect_right(neg, s)
        wins += lo + 0.5 * (hi - lo)
    return wins / (len(pos) * len(neg))

WORD_FIELDS = ["clip", "pitch_band", "onset_band", "form",
               "truth_first", "share_first", "first_token",
               "correct_by_share", "correct_by_word"]
word_rows = []
with open("/content/property_manifest.csv", newline="") as fh:
    prop_rows = [r for r in csv.DictReader(fh)
                 if r["group"] == "tone"]
print(f"property tones to ask: {len(prop_rows)}")
FORMS = [
    ("pitch_reversed", PITCH_REV_Q, "yesno", "low"),
    ("onset_reversed", ONSET_REV_Q, "yesno", "late"),
    ("pitch_choice", PITCH_CHOICE_Q, "choice_pitch", "high"),
    ("onset_choice", ONSET_CHOICE_Q, "choice_onset", "early"),
]
for i, r in enumerate(prop_rows):
    path = os.path.join("/content/property_clips",
                        r["clip_name"] + ".wav")
    truth = {"low": r["pitch_band"] == "low",
             "late": r["onset_band"] == "late",
             "high": r["pitch_band"] == "high",
             "early": r["onset_band"] == "early"}
    for form, question, kind, first_key in FORMS:
        try:
            logits, first = forward_logits(path, question)
        except FeedError as e:
            print()
            print("!" * 70)
            print("PART 2 FEED HARD STOP:")
            print(str(e))
            print(f"Clip {r['clip_name']} was not asked, and "
                  f"the run stops here. Show this whole output "
                  f"to Muse.")
            print("!" * 70)
            sys.exit(7)
        if kind == "yesno":
            share = share_of(logits, YES_TOKEN_IDS, NO_TOKEN_IDS)
            word_first = first.strip().lower().startswith("yes")
        elif kind == "choice_pitch":
            share = share_of(logits, HIGH_IDS, LOW_IDS)
            word_first = first.strip().lower().startswith("high")
        else:
            share = share_of(logits, EARLY_IDS, LATE_IDS)
            word_first = first.strip().lower().startswith("early")
        t = truth[first_key]
        word_rows.append({
            "clip": r["clip_name"],
            "pitch_band": r["pitch_band"],
            "onset_band": r["onset_band"], "form": form,
            "truth_first": t, "share_first": f"{share:.4f}",
            "first_token": first,
            "correct_by_share": (share >= 0.5) == t,
            "correct_by_word": word_first == t})
    if (i + 1) % 60 == 0:
        print(f"  asked {i + 1}/{len(prop_rows)} tones "
              f"(four forms each)")
    free_memory()

def write_word(p):
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=WORD_FIELDS)
        w.writeheader()
        w.writerows(word_rows)
save_both("wording_results.csv", write_word)
print("PART 2 READOUT (mechanical):")
for form, _, _, _ in FORMS:
    rs = [r for r in word_rows if r["form"] == form]
    pairs = [(float(r["share_first"]), r["truth_first"] in (True, "True"))
             for r in rs]
    by_share = sum(r["correct_by_share"] in (True, "True")
                   for r in rs) / len(rs)
    by_word = sum(r["correct_by_word"] in (True, "True")
                  for r in rs) / len(rs)
    print(f"  {form}: by word {by_word:.3f}, by share at 0.5 "
          f"{by_share:.3f}, AUC {auc_of(pairs):.4f} "
          f"({len(rs)} asks)")
print("battery done: the top check cells run and the wording "
      "battery asked on all 360 tones in four forms.")
print("Outputs: topcheck_results.csv, wording_results.csv, "
      "closers_run_log.txt, plus the head's baselines and "
      "token check files.")
'''

compile(RUNNER_SRC, "qwen_closers_run.py", "exec")
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
cells.append(md(r"""# Project K, Series Three: the two closers, Qwen2.5-Omni-3B

**What this is.** The council's follow-up audit (Nov 4, 2026)
left two small items open before the property patch:

- **Part 1, the one-cell top check.** The index account says
  the hidden-state list's last entry is the polished (post-norm)
  final state. Paste a tone's polished state into a silent
  run's polished slot, at the answer position: only the output
  head runs after that slot, so the score must return the
  tone's own baseline exactly. Four self lines, four pastes
  each way.
- **Part 2, the wording battery.** The behavioral result was
  corrected this morning: the model's yes/no lean separates
  pitch and timing almost perfectly, and the wrong answers come
  from where the yes/no line sits. But each question was asked
  one way only. Here the same 360 property tones get the
  questions flipped ("is it low-pitched?", "does it start after
  the first second?") and as direct choices ("high or low?",
  "early or late?"). If the bias lives in the yes/no form, the
  flipped questions mirror it and the direct choices read the
  lean cleanly.

**The machinery is the index-check runner's own head** (proof,
load, scorer, feed, token check, baselines), taken whole.
**Where it runs.** Free Colab account (kadjei1). **How.** Run
the cells from the top, in order.

**The one rule.** We claim only what we ran.
"""))
cells.append(IDX_CELLS[1])   # Step 1 markdown
cells.append(IDX_CELLS[2])   # Step 1 code
cells.append(md(r"""## Step 2 · Get the clips

Two sets, both from the Drive folder `tracing-qwen3b`: the
patching clips (Part 1's tones and silences, with their
manifest) and the property clips with `property_manifest.csv`
(Part 2's 360 tones). The next cell brings the patching clips;
the one after it brings the property files.
"""))
cells.append(IDX_CELLS[4])   # brings patching clips
cells.append(IDX_CELLS[5])   # brings property clips + manifest
cells.append(md(r"""## Step 3 · Write the runner, then check

Check mode runs the proof, the loading candidates, the layer
resolution, the token check, and the feed proof, then stops.
"""))
import re as _re
_write = ''.join(IDX_CELLS[7]["source"])
_write = _write.replace("qwen_indexcheck_run.py", "qwen_closers_run.py")
_write = _re.sub(r"RUNNER_SOURCE = '(?:[^'\\]|\\.)*'",
                 lambda m: "RUNNER_SOURCE = " + repr(RUNNER_SRC),
                 _write, count=1)
cells.append(code(_write))
_check = ''.join(IDX_CELLS[8]["source"]).replace(
    "qwen_indexcheck_run.py", "qwen_closers_run.py")
cells.append(code(_check))
cells.append(md(r"""## Step 4 · Run the battery

Battery mode runs the baselines (the head), then Part 1 (the
top check) and Part 2 (the wording battery). Everything prints
live and is also written to `closers_run_log.txt` in the Drive
folder. Results land in the same folder: `topcheck_results.csv`
and `wording_results.csv`.
"""))
_batt = ''.join(IDX_CELLS[10]["source"]).replace(
    "qwen_indexcheck_run.py", "qwen_closers_run.py")
cells.append(code(_batt))
cells.append(md(r"""## Closing · What to send back

From the Drive folder `tracing-qwen3b`, the run's own words:

1. the PART 1 READOUT block (max diff per direction);
2. the PART 2 READOUT block (by word, by share, AUC per form);
3. the battery exit code;
4. whether `topcheck_results.csv` and `wording_results.csv`
   exist in the folder.
"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3",
                                  "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(HERE, "project_k_qwen_closers_colab.ipynb")
with open(out, "w") as fh:
    json.dump(nb, fh)
print("wrote", out, "cells:", len(cells))
