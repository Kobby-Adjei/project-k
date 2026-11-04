# build_namingcontrol_qwen_notebook.py
# Builds project_k_qwen_namingcontrol_colab.ipynb: the
# both-properties-matched donor control for the open-form
# naming claim — the one blocker that survived the
# council's own test in the Nov 4 freeze audits, run inside
# the freeze gate on Kobby's go.
#
# The gap it closes: in every battery so far, the real and
# sham open-form arms differed in WHICH property was
# crossed (real crosses pitch holding onset; the fixpack
# sham crosses onset holding pitch), never in whether a
# paste happened. So "crossing pitch transplants naming"
# has no no-effect control. This cell adds it: a donor
# matched on BOTH properties (same pitch band, same onset
# band, different instance). If naming follows that paste
# too, the effect is a paste artifact; if it does not, the
# pitch-crossing transplant is real.
#
# Design (one cell, one load — Claude's spec):
#   - The fixpack's 6 low recipients (cells (low, early)
#     and (low, late), instances 0-2 from the manifest).
#   - Arms at layers {14, 19, 24, 28}, all in this run:
#       real:        donor from (high, same onset),
#                    the proppatch selection (offset 1).
#       bothmatched: donor from (low, same onset),
#                    instances 3-5 — same band, same onset,
#                    different instance.
#   - Unpatched answers for every recipient and every
#     donor, recorded in the same run.
#   - Readout: the open question (P3), generated with the
#     surgery's proven recipe (patched_openform).
# Registered prediction: the bothmatched arm stays at the
# recipients' own unpatched answers (no "beep" transplant),
# while the real arm produces "beep" — the pattern that
# would close the naming claim. Either outcome is reported.
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
RUNNER_SRC = RUNNER_SRC.replace("fixpack_", "namingcontrol_")
RUNNER_SRC = RUNNER_SRC.replace("qwen_fixpack_run.py",
                                 "qwen_namingcontrol_run.py")

RUNNER_SRC += r'''
# --- The both-matched naming control ---------------------------------------------
import csv

print("=" * 70)
print("THE BOTH-MATCHED NAMING CONTROL (freeze gate)")
print("=" * 70)
# Preflight: heal the property files from the mounted
# Drive if the Step 2 copies have not landed; exit 13
# with the reason if they cannot be healed.
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

def clip_path(name):
    return os.path.join("/content/property_clips",
                        name + ".wav")

# Triples: (recipient, real donor, both-matched donor).
TRIPLES = []
for ob in ("early", "late"):
    lows = cells_map[("low", ob)]
    highs = cells_map[("high", ob)]
    for i in range(3):
        TRIPLES.append((lows[i], highs[(i + 1) % len(highs)],
                        lows[i + 3]))
print("triples:", len(TRIPLES))
for t in TRIPLES:
    print("  recip %s | real %s | matched %s" % t)

LAYERS = (14, 19, 24, 28)

def openform_unpatched(path):
    inputs, _ = feed_inputs(path, P3, expect_silence=False)
    prefill_len = int(inputs["input_ids"].shape[1])
    with torch.no_grad():
        out = model.generate(
            **inputs, thinker_max_new_tokens=40,
            do_sample=False, return_dict_in_generate=True,
            output_scores=True)
    new_tokens = out.sequences[:, prefill_len:]
    text = processor.batch_decode(
        new_tokens, skip_special_tokens=True,
        clean_up_tokenization_spaces=False)[0].strip()
    del out, new_tokens, inputs
    free_memory()
    return text

rows = []
all_clips = sorted({c for t in TRIPLES for c in t})
for c in all_clips:
    try:
        ans = openform_unpatched(clip_path(c))
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stdout)
        ans = "ERROR (see log)"
    rows.append({"kind": "unpatched", "recipient": c,
                 "donor": "", "layer": "", "answer": ans})
    print("unpatched", c, "->", ans, flush=True)

for recip, real_d, match_d in TRIPLES:
    rp = clip_path(recip)
    for layer in LAYERS:
        for kind, donor in (("real", real_d),
                            ("bothmatched", match_d)):
            try:
                ans = patched_openform(
                    rp, clip_path(donor), layer)
            except Exception:
                import traceback
                traceback.print_exc(file=sys.stdout)
                ans = "ERROR (see log)"
            rows.append({"kind": kind, "recipient": recip,
                         "donor": donor, "layer": layer,
                         "answer": ans})
    print("triple done:", recip, flush=True)

def save_csv(rows, name, fields):
    def _w(p):
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    save_both(name, _w)
    print("saved", name, len(rows), "rows")

save_csv(rows, "namingcontrol_openform.csv",
         ["kind", "recipient", "donor", "layer", "answer"])

print("=" * 70)
print("NAMING CONTROL READOUT")
print("=" * 70)
unp = {r["recipient"]: r["answer"] for r in rows
       if r["kind"] == "unpatched"}
print("unpatched recipients:",
      [unp[t[0]] for t in TRIPLES])
print("unpatched real donors:",
      [unp[t[1]] for t in TRIPLES])
print("unpatched matched donors:",
      [unp[t[2]] for t in TRIPLES])
for layer in LAYERS:
    for kind in ("real", "bothmatched"):
        ans = [r["answer"] for r in rows
               if r["kind"] == kind
               and int(r["layer"]) == layer]
        beeps = sum(1 for a in ans
                    if a.lower().startswith("beep")
                    or a.lower() == "a beep.")
        print("layer %d %s: %d/%d beep-like | %s"
              % (layer, kind, beeps, len(ans), ans))
print("NAMING CONTROL DONE")
'''

cells = []
cells.append(md(r"""# Project K · Qwen 3B · The both-matched naming control (Series Three, freeze gate)

**The question.** When a high donor's states are pasted
into a low recipient, the recipient starts naming the
sound ("beep"). But every control so far crossed a
different property than the real arm did, so the claim has
no true no-effect control. This cell adds it: a donor
matched on BOTH facts (same pitch band, same timing band,
different recording). If the name follows that paste too,
the naming effect is an artifact of pasting itself. If it
does not, crossing pitch is what transplants the name.

**What runs.** Six low recipients, each with a real donor
(high, same timing) and a matched donor (low, same timing,
different recording), pastes at layers 14, 19, 24, 28, and
every clip's unpatched answer, all in one run and one model
load, so every contrast is within-run.

**Where it runs.** Free Colab account (kadjei1). **How.** Run
the cells from the top, in order. Everything prints live and
is also written to `namingcontrol_run_log.txt` in the Drive
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
                        "qwen_namingcontrol_run.py")
_write = re.sub(r"RUNNER_SOURCE = '(?:[^'\\]|\\.)*'",
                lambda m: "RUNNER_SOURCE = " + repr(RUNNER_SRC),
                _write, count=1)
cells.append(code(_write))
_check = ''.join(PROP_CELLS[7]["source"]).replace(
    "qwen_property_run.py", "qwen_namingcontrol_run.py")
cells.append(code(_check))
cells.append(md(r"""## Step 4 · Run the battery

Battery mode records every clip's unpatched answer, then
runs the two paste arms at four layers. Results land in the
Drive folder `tracing-qwen3b`:
`namingcontrol_openform.csv`, `namingcontrol_run_log.txt`.
"""))
_batt = ''.join(PROP_CELLS[9]["source"]).replace(
    "qwen_property_run.py", "qwen_namingcontrol_run.py")
cells.append(code(_batt))
cells.append(md(r"""## Closing · What to send back

From the Drive folder `tracing-qwen3b`, the run's own words:

1. the NAMING CONTROL READOUT block;
2. the battery exit code;
3. whether `namingcontrol_openform.csv` exists in the
   folder.
"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3",
                                  "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(HERE, "project_k_qwen_namingcontrol_colab.ipynb")
with open(out, "w") as fh:
    json.dump(nb, fh)
print("wrote", out, "cells:", len(cells))
