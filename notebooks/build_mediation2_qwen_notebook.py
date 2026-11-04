# build_mediation2_qwen_notebook.py
# Builds project_k_qwen_mediation2_colab.ipynb: the
# corrected mediation (relay) test, per the council's Nov 4
# fixpack audits (Claude's specified cell + ChatGPT's four
# arms), Kobby's go Nov 4, 2026.
#
# The void cell restored the recipient's post-block rows at
# the paste layer only, where those rows are computed from
# pre-paste inputs and are already the recipient's own: it
# tested nothing (96/96 rows bit-identical to audio-only,
# confirmed cross-file by Claude). The corrected cell
# restores the recipient's post-block rows at EVERY layer
# from the paste layer up, where they actually diverge.
#
# Design (both reviewers):
#   - Pairs: the real directional pairs, pitch low_to_high
#     (6) and onset early_to_late (6), from the property
#     manifest cells, the proppatch plan's selection.
#   - Layers {19, 20, 21, 22} (the fade zone), both
#     questions read.
#   - Arms per pair x layer x question:
#       audio:         the paste alone (reference).
#       combined:      paste + post-block rows restored to
#                      the recipient's own states at every
#                      layer from the paste layer up.
#       restore_only:  the same restoration on an UNPATCHED
#                      run. Must return the baseline
#                      bit-for-bit; this is the hook
#                      validation the void cell never had.
#       sham_combined: a gap-matched sham paste (donor
#                      matched on the target property,
#                      full gap on the other) + restoration.
#   - The restored set is the positions after the audio
#     block, the FINAL position excluded (Claude's trap:
#     the final position is itself post-block; restoring it
#     pins the output to the recipient's, a guaranteed
#     null). Its size is printed per cell.
#   - Captures: one unpatched forward per clip per question;
#     donor and recipient states for every layer come from
#     those captures, so all four arms in a cell share the
#     same captured states.
# Registered reading: if combined sits at the recipient's
# baseline while audio sits at the donor's, the post-block
# positions are a necessary route for the paste's effect in
# the fade zone (not the only one; a null means necessary,
# per Claude). If combined tracks audio, the effect does
# not route through them.
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))

_ns = {"__file__": os.path.join(HERE, "build_fixpack_qwen_notebook.py"),
       "__name__": "fixpack_builder"}
exec(open(os.path.join(HERE, "build_fixpack_qwen_notebook.py")).read(), _ns)
FIX_SRC = _ns["RUNNER_SRC"]
PROP_CELLS = _ns["PROP_CELLS"]
CLOSERS_CELLS = _ns["CLOSERS_CELLS"]
code = _ns["code"]
md = _ns["md"]

CUT = FIX_SRC.index("# --- The fixpack plan")
RUNNER_SRC = FIX_SRC[:CUT]
RUNNER_SRC = RUNNER_SRC.replace("fixpack_", "mediation2_")
RUNNER_SRC = RUNNER_SRC.replace("qwen_fixpack_run.py",
                                 "qwen_mediation2_run.py")

RUNNER_SRC += r'''
# --- The corrected mediation test ------------------------------------------------
import csv

print("=" * 70)
print("THE CORRECTED MEDIATION TEST (restore at every layer up)")
print("=" * 70)
# Preflight: heal the property files from the mounted Drive
# if the Step 2 copies have not landed (the proppatch
# lesson); stop with the reason printed, exit 13.
import shutil as _shutil
_need_manifest = not os.path.exists(
    "/content/property_manifest.csv")
_need_clips = (not os.path.isdir("/content/property_clips")
               or len([n for n in os.listdir(
                   "/content/property_clips")
                   if n.endswith(".wav")]) < 400)
if _need_manifest or _need_clips:
    print("preflight: property files missing in this "
          "session; copying from the mounted Drive...")
    try:
        if _need_manifest:
            _shutil.copy(
                "/content/drive/MyDrive/tracing-qwen3b/"
                "property_manifest.csv",
                "/content/property_manifest.csv")
        if _need_clips:
            os.makedirs("/content/property_clips",
                        exist_ok=True)
            _shutil.copytree(
                "/content/drive/MyDrive/tracing-qwen3b/"
                "property_clips",
                "/content/property_clips",
                dirs_exist_ok=True)
        print("preflight: healed from Drive.")
    except Exception as _e:
        print("preflight FAILED:", str(_e)[:200])
        sys.exit(13)
if not os.path.exists("/content/property_manifest.csv"):
    print("preflight: manifest still missing; exit 13.")
    sys.exit(13)
print("preflight: manifest present, %d property clips in "
      "place" % len([n for n in os.listdir(
          "/content/property_clips")
          if n.endswith(".wav")]))

with open("/content/property_manifest.csv") as fh:
    manifest = [r for r in csv.DictReader(fh)
                if r["group"] == "tone"]
cells_map = {}
for r in manifest:
    cells_map.setdefault((r["pitch_band"], r["onset_band"]),
                         []).append(r["clip_name"])
for k in cells_map:
    cells_map[k] = sorted(cells_map[k])
print("cells:", {k: len(v) for k, v in cells_map.items()})

def clip_path(name):
    return os.path.join("/content/property_clips",
                        name + ".wav")

def pick(cell, i):
    return cells_map[cell][i % len(cells_map[cell])]

# Pairs: (direction, recip_cell, donor_cell, sham_cell).
# Sham: matched on the target property, full gap on the
# other (the fixpack rule). Recipients start at index 0,
# donors offset by one, the proppatch plan's selection.
PAIRS = []
for i in range(3):
    for ob in ("early", "late"):
        PAIRS.append(("pitch", "low_to_high",
                      pick(("low", ob), i),
                      pick(("high", ob), i + 1),
                      pick(("low", "late" if ob == "early"
                            else "early"), i + 2)))
for i in range(3):
    for pb in ("low", "high"):
        PAIRS.append(("onset", "early_to_late",
                      pick((pb, "early"), i),
                      pick((pb, "late"), i + 1),
                      pick(("high" if pb == "low" else "low",
                            "early"), i + 2)))
print("pairs:", len(PAIRS))

LAYERS = (19, 20, 21, 22)
QUESTIONS = (("pitch", PITCH_Q), ("onset", ONSET_Q))

def capture_states(path, question):
    """One unpatched forward; returns (inputs, states) with
    states[l] = the hidden-state tuple entry for decoder
    layer l's output (entry l+1), all layers, detached."""
    inputs, _ = feed_inputs(path, question,
                            expect_silence=False)
    out = thinker_forward(inputs)
    states = [out.hidden_states[l + 1][0].detach().clone()
              for l in range(36)]
    del out
    free_memory()
    return inputs, states

def hooked_forward(inputs, layer, audio_state, post_states):
    """One forward with hooks. At the paste layer: audio
    rows take audio_state (if given) and post rows take the
    recipient's own state. At every layer above: post rows
    restored. post_states maps layer -> recipient state.
    Returns (share, first word) under the run's question."""
    pos = audio_block(inputs)
    p_pos = post_positions(inputs)
    prefill_len = int(inputs["input_ids"].shape[1])
    _ref = audio_state
    if _ref is None and post_states:
        _ref = next(iter(post_states.values()))
    _dev = _ref.device if _ref is not None else \
        inputs["input_ids"].device
    a_t = torch.tensor(pos, device=_dev)
    p_t = torch.tensor(p_pos, device=_dev)
    handles = []

    def make_hook(l, with_audio):
        def hook(module, args, output):
            hs = output[0] if isinstance(
                output, tuple) else output
            if hs.shape[1] != prefill_len:
                return output
            new = hs.clone()
            if with_audio and audio_state is not None:
                new[0, a_t] = audio_state[a_t].to(new.dtype)
            rs = post_states.get(l)
            if rs is not None:
                new[0, p_t] = rs[p_t].to(new.dtype)
            if isinstance(output, tuple):
                return (new,) + output[1:]
            return new
        return hook

    for l in range(layer, 36):
        h = THINKER.model.layers[l].register_forward_hook(
            make_hook(l, l == layer))
        handles.append(h)
    try:
        o = thinker_forward(inputs)
    finally:
        for h in handles:
            h.remove()
    share, first = share_and_word(o.logits[0, -1])
    del o
    free_memory()
    return share, first

rows = []
restore_max_diff = 0.0
post_set_size = None
clip_names = sorted({c for p in PAIRS for c in p[2:]})
baselines = {c: margins_for(clip_path(c)) for c in clip_names}
print("baselines read for", len(baselines), "clips")

for arm_dir, direction, recip, donor, sham in PAIRS:
    rp, dp, sp = (clip_path(recip), clip_path(donor),
                  clip_path(sham))
    for qkey, question in QUESTIONS:
        r_in, r_states = capture_states(rp, question)
        d_in, d_states = capture_states(dp, question)
        s_in, s_states = capture_states(sp, question)
        if audio_block(r_in) != audio_block(d_in) or \
                audio_block(r_in) != audio_block(s_in):
            print("PATCH HARD STOP: audio spans differ "
                  "inside a cell.")
            sys.exit(12)
        post_states = {l: r_states[l] for l in range(36)}
        if post_set_size is None:
            post_set_size = len(post_positions(r_in))
            print("restored set: %d post-block positions "
                  "(final position excluded)" % post_set_size)
        rb = baselines[recip][qkey][0]
        db = baselines[donor][qkey][0]
        for layer in LAYERS:
            arms = (
                ("audio", d_states[layer], None),
                ("combined", d_states[layer], post_states),
                ("restore_only", None, post_states),
                ("sham_combined", s_states[layer],
                 post_states))
            for arm, a_state, p_states in arms:
                share, first = hooked_forward(
                    r_in, layer, a_state,
                    p_states if p_states is not None else {})
                if arm == "restore_only":
                    restore_max_diff = max(
                        restore_max_diff, abs(share - rb))
                rows.append({
                    "arm": arm, "direction": direction,
                    "recip": recip, "donor": donor
                    if "sham" not in arm else sham,
                    "layer": layer, "question": qkey,
                    "share": round(share, 4),
                    "first_word": first,
                    "base_share_recip": round(rb, 4),
                    "base_share_donor": round(db, 4)})
        del r_in, d_in, s_in, r_states, d_states, s_states
    print("pair done:", direction, recip, flush=True)

def save_csv(rows, name, fields):
    def _w(p):
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    save_both(name, _w)
    print("saved", name, len(rows), "rows")

save_csv(rows, "mediation2_results.csv",
         ["arm", "direction", "recip", "donor", "layer",
          "question", "share", "first_word",
          "base_share_recip", "base_share_donor"])

print("=" * 70)
print("MEDIATION READOUT")
print("=" * 70)
print("restore_only validation: max |share - baseline| = "
      "%.4f (must be 0.0000)" % restore_max_diff)
for direction in ("low_to_high", "early_to_late"):
    qk = "pitch" if direction == "low_to_high" else "onset"
    for layer in LAYERS:
        line = [direction, "L%d" % layer]
        for arm in ("audio", "combined", "sham_combined"):
            rs = [r for r in rows if r["arm"] == arm
                  and r["direction"] == direction
                  and r["layer"] == layer
                  and r["question"] == qk]
            if rs:
                mv = sum((r["share"] - r["base_share_recip"])
                         * (1 if r["base_share_donor"]
                            > r["base_share_recip"] else -1)
                         for r in rs) / len(rs)
                gap = sum(abs(r["base_share_donor"]
                              - r["base_share_recip"])
                          for r in rs) / len(rs)
                line.append("%s %+.3f (gap %.3f)"
                            % (arm, mv, gap))
        print(" | ".join(line))
print("MEDIATION DONE")
'''

cells = []
cells.append(md(r"""# Project K · Qwen 3B · The corrected mediation test (Series Three)

**The question.** In the fade zone (layers 19 to 22), does
the paste's effect on the answer travel THROUGH the text
positions after the sound block? The first mediation cell
was void: it restored those positions at the paste layer
only, where they are computed from pre-paste inputs and are
already the recipient's own, so all 96 rows came out
bit-identical to the plain paste. This version restores
them at every layer from the paste layer up, where they
actually diverge.

**What runs.** Twelve real pairs (pitch low-to-high,
timing early-to-late), layers 19 to 22, both questions,
four arms: the paste alone; the paste with the later
positions held to the recipient's own states at every layer
up; the same holding on an unpatched run, which must return
the baseline exactly or the hooks are not proven; and a
gap-matched sham paste with the same holding. The final
answer position is never restored: it is itself a later
position, and restoring it would pin the answer by
construction.

**Where it runs.** Free Colab account (kadjei1). **How.** Run
the cells from the top, in order. Everything prints live and
is also written to `mediation2_run_log.txt` in the Drive
folder `tracing-qwen3b`.

**The one rule.** We claim only what we ran.
"""))
cells.append(PROP_CELLS[1])   # Step 1 markdown
cells.append(PROP_CELLS[2])   # Step 1 code
cells.append(md(r"""## Step 2 · Get the clips and the masters

The battery needs the step-5 property files (clips plus
`property_manifest.csv`) from the Drive folder
`tracing-qwen3b`, and the master sounds for the head's token
check. The next cell brings the masters; the one after brings
the property files. If the copies have not landed in a
session, the runner heals them from the mounted Drive and
says so in the log.
"""))
cells.append(PROP_CELLS[4])       # masters for the head
cells.append(CLOSERS_CELLS[5])    # property clips + manifest
cells.append(md(r"""## Step 3 · Write the runner, then check

Check mode runs the proof, the loading candidates, and the
token check, then stops.
"""))
_write = ''.join(PROP_CELLS[6]["source"])
_write = _write.replace("qwen_property_run.py",
                        "qwen_mediation2_run.py")
_write = re.sub(r"RUNNER_SOURCE = '(?:[^'\\]|\\.)*'",
                lambda m: "RUNNER_SOURCE = " + repr(RUNNER_SRC),
                _write, count=1)
cells.append(code(_write))
_check = ''.join(PROP_CELLS[7]["source"]).replace(
    "qwen_property_run.py", "qwen_mediation2_run.py")
cells.append(code(_check))
cells.append(md(r"""## Step 4 · Run the battery

Battery mode reads baselines for every clip in the pair
plan, then runs the four arms. Results land in the Drive
folder `tracing-qwen3b`: `mediation2_results.csv`,
`mediation2_run_log.txt`.
"""))
_batt = ''.join(PROP_CELLS[9]["source"]).replace(
    "qwen_property_run.py", "qwen_mediation2_run.py")
cells.append(code(_batt))
cells.append(md(r"""## Closing · What to send back

From the Drive folder `tracing-qwen3b`, the run's own words:

1. the MEDIATION READOUT block (including the restore_only
   validation line);
2. the battery exit code;
3. whether `mediation2_results.csv` exists in the folder.
"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3",
                                  "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(HERE, "project_k_qwen_mediation2_colab.ipynb")
with open(out, "w") as fh:
    json.dump(nb, fh)
print("wrote", out, "cells:", len(cells))
