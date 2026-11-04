# build_proppatch_qwen_notebook.py
# Builds project_k_qwen_proppatch_colab.ipynb: the property
# patch, the causal step in the council's order (Nov 4, 2026;
# Kobby's go after the closers).
# Design (council spec: ChatGPT's one-property-at-a-time with
# independent donors and sham controls; Claude's partial-layer
# open-form cells; the closers' readout ruling):
#   - Stimuli: the step-5 property tones (360 factorial clips).
#     Recipients and donors are independent instances, matched
#     on the OTHER property, opposed on the patched one.
#   - Patch site: the audio positions only, one decoder layer
#     at a time, grid {0, 14, 16, 18, 19, 20, 22, 24, 28, 34}
#     (34 is the deepest clean layer; 35 is excluded as the
#     known double-norm artifact). Donor states are captured
#     under the same question as the readout.
#   - Readouts, both kept: the normalized yes share (margin)
#     and the emitted first word, under the ORIGINAL questions
#     (the wording battery showed their margins rank at AUC
#     1.0000 / 0.9988; the reversed pitch wording degrades its
#     margin and is not used). Each arm also reads the OTHER
#     property's margin as a specificity control.
#   - Controls: sham patches (donor carries the SAME property
#     value, different instance; the margin should not move)
#     and post-position patches (states after the audio block,
#     final position excluded, at layers {0, 19, 28}; a strict
#     subset that cannot assemble the hybrid state).
#   - Open form at partial layers (Claude): the pitch arm's
#     low-to-high pairs at layers {14, 19, 24} also get the
#     open question under the patch.
# The runner head is the property battery's own head (proof,
# load, feed, token check), taken whole up to clip generation;
# the property clips already exist in the Drive folder.
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))

_ns = {"__file__": os.path.join(HERE, "build_property_qwen_notebook.py"),
       "__name__": "property_builder"}
exec(open(os.path.join(HERE, "build_property_qwen_notebook.py")).read(), _ns)
PROP_SRC = _ns["RUNNER_SRC"]
PROP_CELLS = _ns["cells"]

_ns2 = {"__file__": os.path.join(HERE, "build_closers_qwen_notebook.py"),
        "__name__": "closers_builder"}
exec(open(os.path.join(HERE, "build_closers_qwen_notebook.py")).read(), _ns2)
CLOSERS_CELLS = _ns2["cells"]

CUT = PROP_SRC.index("# --- Probe clip generation")
head = PROP_SRC[:CUT]
head = head.replace("property_", "proppatch_")
head = head.replace("qwen_property_run.py", "qwen_proppatch_run.py")

RUNNER_SRC = head

RUNNER_SRC += r'''
import csv  # the property head keeps its csv import in the
# probe tail this runner cuts off; the first version of this
# battery died on exactly that (NameError after the preflight
# print, Nov 4, 2026). Imports the appended parts need live
# here, not in the borrowed head.

# --- Scorers and questions -------------------------------------------------------
# The original behavioral questions (index check Part B); the
# wording battery (Nov 4, 2026) showed their margins rank the
# properties at AUC 1.0000 (pitch) and 0.9988 (onset), so they
# are the readouts. The reversed pitch wording degraded its
# own margin (0.6331) and is not used.
YES_TOKEN_IDS = []
for _word in ("Yes", " Yes"):
    _enc = processor.tokenizer.encode(_word,
                                      add_special_tokens=False)
    if _enc:
        YES_TOKEN_IDS.append(int(_enc[0]))
YES_TOKEN_IDS = sorted(set(YES_TOKEN_IDS))
NO_TOKEN_IDS = []
for _word in ("No", " No"):
    _enc = processor.tokenizer.encode(_word,
                                      add_special_tokens=False)
    if _enc:
        NO_TOKEN_IDS.append(int(_enc[0]))
NO_TOKEN_IDS = sorted(set(NO_TOKEN_IDS))
PITCH_Q = ("Is the tone in this clip high-pitched? "
           "Answer yes or no.")
ONSET_Q = ("Does the tone start in the first second of this "
           "clip? Answer yes or no.")
P3 = ("What sounds can you hear in this clip? Describe only "
      "what is audible, without guessing from the pictures. "
      "If the audio is silent, say 'silent'.")
print("scorer tokens: yes", YES_TOKEN_IDS, "no", NO_TOKEN_IDS)

def share_and_word(logits):
    probs = torch.softmax(logits.float(), dim=-1)
    py = sum(float(probs[i].item()) for i in YES_TOKEN_IDS)
    pn = sum(float(probs[i].item()) for i in NO_TOKEN_IDS)
    share = py / (py + pn) if (py + pn) > 0 else 0.0
    first = processor.tokenizer.decode(
        [int(torch.argmax(logits).item())])
    return share, first

def margins_for(path):
    """Both questions' margins and words, unpatched."""
    out = {}
    for key, q in (("pitch", PITCH_Q), ("onset", ONSET_Q)):
        inputs, _ = feed_inputs(path, q, expect_silence=False)
        o = thinker_forward(inputs)
        share, first = share_and_word(o.logits[0, -1])
        out[key] = (share, first)
        del o, inputs
        free_memory()
    return out

def audio_block(inputs):
    ids = inputs["input_ids"][0].tolist()
    pos = [i for i, t in enumerate(ids) if t == AUDIO_TOKEN_ID]
    if not pos:
        print("PATCH HARD STOP: no audio positions in a feed.")
        sys.exit(12)
    if pos != list(range(pos[0], pos[-1] + 1)):
        print("PATCH HARD STOP: audio positions not contiguous.")
        sys.exit(12)
    return pos

def post_positions(inputs):
    pos = audio_block(inputs)
    last = int(inputs["input_ids"].shape[1]) - 1
    return list(range(pos[-1] + 1, last))

def patched_share(recip_path, donor_path, layer, mode,
                  question):
    """One patched forward: the donor's state at decoder
    layer `layer` (tuple index layer+1) replaces the
    recipient's rows at the mode's positions, during the
    prefill pass only. Returns the question's yes share and
    first word from the patched logits."""
    r_in, _ = feed_inputs(recip_path, question,
                          expect_silence=False)
    d_in, _ = feed_inputs(donor_path, question,
                          expect_silence=False)
    if int(r_in["input_ids"].shape[1]) != int(
            d_in["input_ids"].shape[1]):
        print("PATCH HARD STOP: donor and recipient rows "
              "differ in length.")
        sys.exit(12)
    d_out = thinker_forward(d_in)
    d_state = d_out.hidden_states[layer + 1][0].detach().clone()
    pos = (audio_block(r_in) if mode == "audio"
           else post_positions(r_in))
    pos_t = torch.tensor(pos, device=d_state.device)
    prefill_len = int(r_in["input_ids"].shape[1])
    layer_mod = THINKER.model.layers[layer]

    def hook(module, args, output):
        hs = output[0] if isinstance(output, tuple) else output
        if hs.shape[1] != prefill_len:
            return output
        new = hs.clone()
        new[0, pos_t] = d_state[pos_t].to(new.dtype)
        if isinstance(output, tuple):
            return (new,) + output[1:]
        return new

    handle = layer_mod.register_forward_hook(hook)
    try:
        o = thinker_forward(r_in)
    finally:
        handle.remove()
    share, first = share_and_word(o.logits[0, -1])
    del o, r_in, d_in, d_out, d_state
    free_memory()
    return share, first

def patched_openform(recip_path, donor_path, layer):
    """The open question (P3) generated with the audio-position
    patch held through the prefill pass."""
    r_in, _ = feed_inputs(recip_path, P3, expect_silence=False)
    d_in, _ = feed_inputs(donor_path, P3, expect_silence=False)
    d_out = thinker_forward(d_in)
    d_state = d_out.hidden_states[layer + 1][0].detach().clone()
    pos = audio_block(r_in)
    pos_t = torch.tensor(pos, device=d_state.device)
    prefill_len = int(r_in["input_ids"].shape[1])
    layer_mod = THINKER.model.layers[layer]

    def hook(module, args, output):
        hs = output[0] if isinstance(output, tuple) else output
        if hs.shape[1] != prefill_len:
            return output
        new = hs.clone()
        new[0, pos_t] = d_state[pos_t].to(new.dtype)
        if isinstance(output, tuple):
            return (new,) + output[1:]
        return new

    handle = layer_mod.register_forward_hook(hook)
    try:
        # The surgery's proven recipe, copied exactly (the
        # handover rerun's generate_from_inputs): the FULL
        # model's generate, no_grad, dict output, batch_decode.
        # This function's first version called
        # THINKER.generate with a bare tensor decode and
        # failed on all 18 cells (Nov 4, 2026); the thinker
        # submodule is not the generation entry point the
        # surgery used.
        with torch.no_grad():
            out = model.generate(
                **r_in, thinker_max_new_tokens=40,
                do_sample=False, return_dict_in_generate=True,
                output_scores=True)
    finally:
        handle.remove()
    new_tokens = out.sequences[:, prefill_len:]
    text = processor.batch_decode(
        new_tokens, skip_special_tokens=True,
        clean_up_tokenization_spaces=False)[0].strip()
    del out, new_tokens, r_in, d_in, d_out, d_state
    free_memory()
    return text

'''

RUNNER_SRC += r'''
# --- The pair plan -----------------------------------------------------------------
# Deterministic from the step-5 manifest. Cells are
# (pitch_band, onset_band); pairing is by sorted clip name,
# donors offset by one inside the matched cell, so a donor is
# never its recipient's own clip.
print("=" * 70)
print("THE PROPERTY PATCH")
print("=" * 70)
# Preflight (added after the first battery attempt died at
# this exact spot with no trace in the log, Nov 4, 2026: the
# Step 2 copies had not landed in that session). The battery
# heals itself from the mounted Drive if it can, and says so
# in the log either way; if it cannot, it stops here with the
# reason printed, exit 13.
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
            _src = ("/content/drive/MyDrive/tracing-qwen3b/"
                    "property_clips")
            for _n in os.listdir(_src):
                if _n.endswith(".wav") and not os.path.exists(
                        os.path.join("/content/property_clips",
                                     _n)):
                    _shutil.copy(os.path.join(_src, _n),
                                 "/content/property_clips")
        print("preflight: healed from Drive.")
    except Exception as _e:
        print("!" * 70)
        print("PREFLIGHT STOP: the property files are not in "
              "place and the Drive copy failed:")
        print(repr(_e))
        print("Step 2 of the notebook did not complete in "
              "this session. Show this log to Muse.")
        print("!" * 70)
        sys.exit(13)
_nwavs = len([n for n in os.listdir("/content/property_clips")
              if n.endswith(".wav")])
print(f"preflight: manifest present, {_nwavs} property "
      f"clips in place")
with open("/content/property_manifest.csv", newline="") as fh:
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

LAYERS = [0, 14, 16, 18, 19, 20, 22, 24, 28, 34]
POST_LAYERS = [0, 19, 28]
OPEN_LAYERS = [14, 19, 24]
PAIRS_PER_CELL = 3

plan = []   # (arm, direction, mode, recip, donor)
def add_pairs(arm, direction, mode, recip_cell, donor_cell,
              n=PAIRS_PER_CELL):
    recips = cells_map[recip_cell][:n]
    donors = cells_map[donor_cell]
    for i, recip in enumerate(recips):
        donor = donors[(i + 1) % len(donors)]
        if donor == recip:
            donor = donors[(i + 2) % len(donors)]
        plan.append((arm, direction, mode, recip, donor))

for onset in ("early", "late"):
    add_pairs("pitch", "low_to_high", "audio",
              ("low", onset), ("high", onset))
    add_pairs("pitch", "high_to_low", "audio",
              ("high", onset), ("low", onset))
    add_pairs("pitch", "sham_low", "audio",
              ("low", onset), ("low", onset))
    add_pairs("pitch", "low_to_high", "post",
              ("low", onset), ("high", onset))
for pitch in ("low", "high"):
    add_pairs("onset", "early_to_late", "audio",
              (pitch, "early"), (pitch, "late"))
    add_pairs("onset", "late_to_early", "audio",
              (pitch, "late"), (pitch, "early"))
    add_pairs("onset", "sham_early", "audio",
              (pitch, "early"), (pitch, "early"))
    add_pairs("onset", "early_to_late", "post",
              (pitch, "early"), (pitch, "late"))
print(f"pair plan: {len(plan)} pairs")

# Baselines for every clip the plan touches.
names = sorted({p[3] for p in plan} | {p[4] for p in plan})
baselines = {}
truth = {r["clip_name"]: (r["pitch_band"], r["onset_band"])
         for r in manifest}
for name in names:
    baselines[name] = margins_for(clip_path(name))
print(f"baselines read for {len(baselines)} clips")

PP_FIELDS = ["arm", "direction", "mode", "layer", "recipient",
             "donor", "question", "base_share",
             "donor_base_share", "share_after", "first_word",
             "word_matches_donor"]
pp_rows = []

def run_cell(arm, direction, mode, layer, recip, donor):
    for qkey, question in (("pitch", PITCH_Q),
                           ("onset", ONSET_Q)):
        share, first = patched_share(
            clip_path(recip), clip_path(donor), layer, mode,
            question)
        donor_val = (truth[donor][0] == "high" if qkey == "pitch"
                     else truth[donor][1] == "early")
        word_yes = first.strip().lower().startswith("yes")
        pp_rows.append({
            "arm": arm, "direction": direction, "mode": mode,
            "layer": layer, "recipient": recip, "donor": donor,
            "question": qkey,
            "base_share": f"{baselines[recip][qkey][0]:.4f}",
            "donor_base_share":
                f"{baselines[donor][qkey][0]:.4f}",
            "share_after": f"{share:.4f}",
            "first_word": first,
            "word_matches_donor": word_yes == donor_val})

total_cells = 0
for arm, direction, mode, recip, donor in plan:
    layers = POST_LAYERS if mode == "post" else LAYERS
    for layer in layers:
        run_cell(arm, direction, mode, layer, recip, donor)
        total_cells += 1
    print(f"  done {arm}/{direction}/{mode}: {recip} <- {donor}")

def write_pp(p):
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=PP_FIELDS)
        w.writeheader()
        w.writerows(pp_rows)
save_both("proppatch_results.csv", write_pp)
print(f"patch cells run: {total_cells} (two questions each)")

# Open form at the partial layers, pitch arm, low to high.
OF_FIELDS = ["recipient", "donor", "layer", "answer"]
of_rows = []
for arm, direction, mode, recip, donor in plan:
    if not (arm == "pitch" and direction == "low_to_high"
            and mode == "audio"):
        continue
    for layer in OPEN_LAYERS:
        try:
            text = patched_openform(clip_path(recip),
                                    clip_path(donor), layer)
        except Exception as _e:
            # Never let one generation kill the battery again
            # (Nov 4, 2026): the traceback goes to the LOG (file
            # must be sys.stdout; the default stderr is not
            # teed, which is why the first failure left no
            # trace) and the row itself carries the reason.
            import traceback as _tb
            _tb.print_exc(file=sys.stdout)
            text = f"ERROR {_e.__class__.__name__}: {_e}"[:200]
        of_rows.append({"recipient": recip, "donor": donor,
                        "layer": layer, "answer": text})
        print(f"  open form {recip} <- {donor} @ {layer}: "
              f"{text!r}")

def write_of(p):
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=OF_FIELDS)
        w.writeheader()
        w.writerows(of_rows)
save_both("proppatch_openform.csv", write_of)

print("PROPPATCH READOUT (mechanical):")
for arm in ("pitch", "onset"):
    for direction in sorted({p[1] for p in plan if p[0] == arm}):
        for mode in ("audio", "post"):
            rs = [r for r in pp_rows
                  if r["arm"] == arm
                  and r["direction"] == direction
                  and r["mode"] == mode
                  and r["question"] == arm]
            if not rs:
                continue
            line = []
            for layer in sorted({int(r["layer"]) for r in rs}):
                sel = [r for r in rs
                       if int(r["layer"]) == layer]
                moved = sum(
                    abs(float(r["share_after"])
                        - float(r["base_share"]))
                    for r in sel) / len(sel)
                toward = sum(
                    (float(r["share_after"])
                     - float(r["base_share"]))
                    * (1 if float(r["donor_base_share"])
                       > float(r["base_share"]) else -1)
                    for r in sel) / len(sel)
                line.append(f"L{layer}: move {moved:.3f} "
                            f"toward {toward:+.3f}")
            print(f"  {arm}/{direction}/{mode} (own question): "
                  + "; ".join(line))
print("battery done: the property patch cells ran (audio, "
      "sham, and post-position controls), with open-form cells "
      "at the partial layers.")
print("Outputs: proppatch_results.csv, "
      "proppatch_openform.csv, proppatch_run_log.txt, plus "
      "the head's token check files.")
'''

compile(RUNNER_SRC, "qwen_proppatch_run.py", "exec")
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
cells.append(md(r"""# Project K, Series Three: the property patch, Qwen2.5-Omni-3B

**What this is.** The causal step the council ordered. The
probes say pitch and timing stay readable at every layer; the
behavioral work says the model's own lean ranks them almost
perfectly while its spoken answers default wrong. This battery
asks the causal question: if we paste a donor tone's insides
into a different tone's run, at one layer, does the lean move
toward the donor's property?

- **Arms.** Pitch (low recipient, high donor, and the reverse)
  and onset (early recipient, late donor, and the reverse).
  Donors are independent clips, matched on the other property.
- **Patch site.** The sound's own positions, one layer at a
  time: 0, 14, 16, 18, 19, 20, 22, 24, 28, 34. (Layer 35 is
  left out on purpose: the index check showed its state is the
  polished one, and pasting it double-polishes.)
- **Readouts.** Both, under the original questions: the yes
  share (the lean) and the first spoken word. Each arm also
  reads the other property's lean, which should NOT move if
  the patch is specific.
- **Controls.** Sham pastes (donor with the SAME property
  value: the lean should stay) and post-position pastes (the
  text positions after the sound, at layers 0, 19, 28: a strict
  subset that cannot rebuild the hybrid state).
- **Open form.** The pitch arm's low-to-high pairs also get
  the open question at layers 14, 19, 24, the partial band.

**Where it runs.** Free Colab account (kadjei1). **How.** Run
the cells from the top, in order. Everything prints live and
is also written to `proppatch_run_log.txt` in the Drive
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
the property files.
"""))
cells.append(PROP_CELLS[4])       # masters for the head
cells.append(CLOSERS_CELLS[5])    # property clips + manifest
cells.append(md(r"""## Step 3 · Write the runner, then check

Check mode runs the proof, the loading candidates, and the
token check, then stops.
"""))
_write = ''.join(PROP_CELLS[6]["source"])
_write = _write.replace("qwen_property_run.py",
                        "qwen_proppatch_run.py")
_write = re.sub(r"RUNNER_SOURCE = '(?:[^'\\]|\\.)*'",
                lambda m: "RUNNER_SOURCE = " + repr(RUNNER_SRC),
                _write, count=1)
cells.append(code(_write))
_check = ''.join(PROP_CELLS[7]["source"]).replace(
    "qwen_property_run.py", "qwen_proppatch_run.py")
cells.append(code(_check))
cells.append(md(r"""## Step 4 · Run the battery

Battery mode reads baselines for every clip in the pair plan,
then runs the patch cells (two questions each), then the
open-form cells. A full run is long (roughly half an hour or
more). Results land in the Drive folder `tracing-qwen3b`:
`proppatch_results.csv`, `proppatch_openform.csv`,
`proppatch_run_log.txt`.
"""))
_batt = ''.join(PROP_CELLS[9]["source"]).replace(
    "qwen_property_run.py", "qwen_proppatch_run.py")
cells.append(code(_batt))
cells.append(md(r"""## Closing · What to send back

From the Drive folder `tracing-qwen3b`, the run's own words:

1. the PROPPATCH READOUT block;
2. the battery exit code;
3. whether `proppatch_results.csv` and
   `proppatch_openform.csv` exist in the folder.
"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3",
                                  "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(HERE, "project_k_qwen_proppatch_colab.ipynb")
with open(out, "w") as fh:
    json.dump(nb, fh)
print("wrote", out, "cells:", len(cells))
