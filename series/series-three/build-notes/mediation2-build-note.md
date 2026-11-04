# The corrected mediation test: build note (Nov 4, 2026)

**What it is.** The council's corrected relay cell (Claude's
design + ChatGPT's four arms), built Nov 4, 2026, runs after
the probe cell on Kobby's order. The first mediation cell
was void by construction (restored post-block rows at the
paste layer only, where they are already the recipient's
own; 96/96 rows bit-identical to audio-only, confirmed
cross-file by Claude's audit). This version restores the
recipient's post-block rows at EVERY layer from the paste
layer up, where they actually diverge.

**Design.** 12 real pairs (pitch low_to_high 6, onset
early_to_late 6; the proppatch plan's selection from the
manifest cells), layers {19, 20, 21, 22}, both questions.
Arms: audio (paste alone); combined (paste + restore);
restore_only (restore on an unpatched run; must return the
baseline bit-for-bit — the hook validation the void cell
never had; the readout prints the max difference);
sham_combined (gap-matched sham paste + restore). The
restored set is the positions after the audio block with
the FINAL position excluded (restoring it would pin the
output by construction); its size prints in the log. One
capture per clip per question feeds all arms in a cell, so
arms share captured states exactly.

**Registered reading.** Combined at the recipient's
baseline while audio sits at the donor's: the post-block
positions are a necessary route in the fade zone (not the
only one). Combined tracking audio: the effect does not
route through them.

**Builder.** `colab/build_mediation2_qwen_notebook.py`
(fixpack head + machinery whole, battery replaced; the
preflight self-heals the property files from Drive, exit
13 with the reason if it cannot; index tensors placed on
the captured states' device). Notebook
`project_k_qwen_mediation2_colab.ipynb`, 12 cells. Outputs:
mediation2_results.csv, mediation2_run_log.txt in the
tracing-qwen3b Drive folder.
