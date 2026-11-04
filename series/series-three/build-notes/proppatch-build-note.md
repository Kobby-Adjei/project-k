# The property patch: build note (Nov 4, 2026)

Built on Kobby's go after the two closers. Notebook:
colab/project_k_qwen_proppatch_colab.ipynb (Drive, folder
tracing-qwen3b); builder: colab/build_proppatch_qwen_notebook.py.
Runner (32,583 chars, compiles; embedded copy extracted and
compile-checked) = the property battery's head whole (proof,
load, feed, token check) up to clip generation, outputs renamed
property_ -> proppatch_, plus the patch machinery.

Design. Stimuli: the step-5 property tones. Pair plan is
deterministic from the manifest: 48 pairs. Pitch arm per onset
band: low_to_high, high_to_low, sham_low (same-value donor,
different instance), and a post-position low_to_high; onset arm
mirrors per pitch band (early_to_late, late_to_early,
sham_early, post). 3 pairs per cell. Patch site: audio
positions, one decoder layer at a time, grid {0, 14, 16, 18,
19, 20, 22, 24, 28, 34}; layer 35 excluded by design (the
double-norm artifact). Post-position controls at {0, 19, 28}
patch the text positions after the audio block (final position
excluded), a strict subset that cannot assemble the hybrid
state. Donor states are captured under the same question as
the readout; the hook rewrites rows only during the prefill
pass (sequence-length guard). Readouts: the original
questions' yes share and first word, both kept; each cell also
reads the OTHER property's margin as a specificity control.
Open form (Claude's cell): pitch low_to_high pairs at layers
{14, 19, 24} get P3 under the patch. Outputs:
proppatch_results.csv (patch cells x 2 questions),
proppatch_openform.csv, proppatch_run_log.txt. Estimated run
length: roughly 30-40 minutes on the free lane (about 1,650
forwards plus 18 generations).

Registered predictions: if the property is causally carried at
the audio positions the way presence was, the recipient's
margin moves toward the donor's baseline in a layer band and
the sham stays near zero move; the other property's margin
does not move (specificity); the post-position control does
not move the margin. Where the band sits for properties
(early like presence, or later) is the open question the
battery answers.

Attempt log. Attempt 1 (Nov 4, 2026, ~5:33 AM): battery exited 1
about 24 s in, right after the battery header; the Step 2 copies
had not landed in that session (diagnosed from the log's shape;
the pair-plan code itself tested clean locally, 48 pairs). Fix:
a self-healing preflight in the runner (copies from the mounted
Drive, prints what it did, exit 13 with the reason if it cannot).
Attempt 2 (~5:59 AM): Step 2 verified cell by cell (420 clips,
manifest 99,543 bytes), preflight passed ("manifest present, 420
property clips in place"), then exit 1 again at ~25 s. Root
cause found by inspection: the property runner keeps
"import csv" in its probe tail, which this runner cuts off; the
appended parts use csv and died with a NameError that only
reached stderr. (The run report's extra "preflight: healing..."
lines exist nowhere in the runner or the log; narration garble,
discarded.) Fix: the appended parts carry their own imports,
and a static audit now checks every name the appended parts
load against the head plus the parts themselves.

Attempt 3 (~6:13 AM): all 396 patch cells ran and
proppatch_results.csv saved (792 rows), then the battery died
at the first patched_openform generation call (traceback again
stderr-only). Fixes applied to the canonical builder: the
generate call is simplified (thinker_max_new_tokens only, no
None kwargs, per the Series Two lesson) and every open-form
cell is guarded (traceback printed into the log, row records
"ERROR (see log)"). The open-form cells rerun alone in
project_k_qwen_proppatch_openform_colab.ipynb
(colab/build_proppatch_openform_notebook.py), which reuses this
runner's head and machinery with its own log name.

Open-form resolution (Nov 4, 2026): the first OF-only run
errored on all 18 cells and the guard's traceback went to
stderr (print_exc without a file argument), leaving no trace
— the guard itself was fixed to print_exc(file=sys.stdout)
with the exception embedded in the row. The generation failure's
cause: patched_openform called THINKER.generate; every working
battery in the project calls the full model's generate. The
function now copies the handover rerun's generate_from_inputs
recipe exactly (model.generate under no_grad,
return_dict_in_generate, batch_decode). Rerun clean: 18/18
answers "beep", exit 0.
