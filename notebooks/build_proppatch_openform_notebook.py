# build_proppatch_openform_notebook.py
# Builds project_k_qwen_proppatch_openform_colab.ipynb: the
# open-form cells of the property patch, alone. The main
# battery (Nov 4, 2026) ran all 396 patch cells and saved
# proppatch_results.csv, then died at the first generation
# call. This variant reuses the proppatch runner's head and
# machinery (with the generation call simplified and each
# cell guarded), rebuilds the same deterministic pair plan,
# and runs only the pitch low-to-high open-form cells at
# layers {14, 19, 24}. Output: proppatch_openform.csv; log:
# proppatch_openform_log.txt.
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))

_ns = {"__file__": os.path.join(HERE, "build_proppatch_qwen_notebook.py"),
       "__name__": "proppatch_builder"}
exec(open(os.path.join(HERE, "build_proppatch_qwen_notebook.py")).read(), _ns)
FULL_SRC = _ns["RUNNER_SRC"]
PP_CELLS = _ns["cells"]

plan_at = FULL_SRC.index("# --- The pair plan")
plan_end = FULL_SRC.index('print(f"pair plan: {len(plan)} pairs")')
plan_end = FULL_SRC.index("\n", plan_end) + 1
of_at = FULL_SRC.index("# Open form at the partial layers")
of_end = FULL_SRC.index('print("PROPPATCH READOUT (mechanical):")')

RUNNER_SRC = FULL_SRC[:plan_at] + FULL_SRC[plan_at:plan_end] + r'''
print("open-form-only variant: the patch cells are already "
      "saved in proppatch_results.csv; this run does the "
      "open-form cells only.")

''' + FULL_SRC[of_at:of_end] + r'''
print("battery done: the open-form cells ran (pitch "
      "low-to-high at layers 14, 19, 24).")
print("Outputs: proppatch_openform.csv, "
      "proppatch_openform_log.txt.")
'''
RUNNER_SRC = RUNNER_SRC.replace("proppatch_run_log.txt",
                                "proppatch_openform_log.txt")
RUNNER_SRC = RUNNER_SRC.replace("qwen_proppatch_run.py",
                                "qwen_proppatch_of_run.py")
compile(RUNNER_SRC, "qwen_proppatch_of_run.py", "exec")
print("runner compiles, chars:", len(RUNNER_SRC))

def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.splitlines(keepends=True)}

def code(text):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}

cells = []
cells.append(md(r"""# Project K, Series Three: the property patch, open-form cells only

The main property-patch battery ran all 396 patch cells and
saved `proppatch_results.csv`, then stopped at the first
generation call. This notebook runs only the missing piece:
the open question ("What sounds can you hear in this clip?")
on the pitch arm's low-to-high pairs, with the donor's states
pasted at the sound's positions at layers 14, 19, and 24, the
partial band. The generation call is simplified and every
cell is guarded, so a failure prints its reason into the log
(`proppatch_openform_log.txt`) instead of ending the run.

**Where it runs.** Free Colab account (kadjei1). **How.** Run
the cells from the top, in order: Step 1, then each Step 2
cell one at a time (read their printed confirmations), then
Run all for Steps 3 and 4.

**The one rule.** We claim only what we ran.
"""))
cells.append(PP_CELLS[1])
cells.append(PP_CELLS[2])
cells.append(PP_CELLS[3])
cells.append(PP_CELLS[4])
cells.append(PP_CELLS[5])
cells.append(md(r"""## Step 3 · Write the runner, then check

Check mode runs the proof, the loading candidates, and the
token check, then stops.
"""))
_write = ''.join(PP_CELLS[7]["source"])
_write = _write.replace("qwen_proppatch_run.py",
                        "qwen_proppatch_of_run.py")
_write = re.sub(r"RUNNER_SOURCE = '(?:[^'\\]|\\.)*'",
                lambda m: "RUNNER_SOURCE = " + repr(RUNNER_SRC),
                _write, count=1)
cells.append(code(_write))
_check = ''.join(PP_CELLS[8]["source"]).replace(
    "qwen_proppatch_run.py", "qwen_proppatch_of_run.py")
cells.append(code(_check))
cells.append(md(r"""## Step 4 · Run the open-form cells

Battery mode rebuilds the pair plan, then runs the 18
open-form cells. Results land in the Drive folder
`tracing-qwen3b`: `proppatch_openform.csv` and
`proppatch_openform_log.txt`.
"""))
_batt = ''.join(PP_CELLS[10]["source"]).replace(
    "qwen_proppatch_run.py", "qwen_proppatch_of_run.py")
cells.append(code(_batt))
cells.append(md(r"""## Closing · What to send back

From the Drive folder `tracing-qwen3b`, the run's own words:

1. the 18 "open form" lines (recipient, donor, layer, answer);
2. the battery exit code;
3. whether `proppatch_openform.csv` exists in the folder.
"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3",
                                  "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(HERE, "project_k_qwen_proppatch_openform_colab.ipynb")
with open(out, "w") as fh:
    json.dump(nb, fh)
print("wrote", out, "cells:", len(cells))
