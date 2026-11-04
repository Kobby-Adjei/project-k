# The probe cell: build note (Nov 4, 2026)

**What it is.** Claude's separating cell from the Nov 4
fixpack audits, run at Kobby's instruction ("Run the probe
cell, then the corrected relay test"). The regression cell
(the audits' other no-run check) already landed from the
behavioral file: within-band, the pitch lean does not track
exact frequency (corr 0.001 low / 0.014 high, n=180 each),
while the onset lean tracks exact start time within band
(-0.500 / -0.338). So the lean is a band meter for pitch.
This cell asks whether the pasted STATE is finer than the
lean: paste a wide within-band pitch donor, then read the
patched run's downstream states with the graded probe
(ridge on PCA-64, the graded battery's readout, trained
in-run on 96 fresh tones, 24 log-spaced freqs 300-2400 Hz x
4 onsets, states under the pitch question).

**Pairs.** (400, 900) Hz and (1600, 2200) Hz, both
directions, 2 instances each = 8 pairs; onset matched at
1.5 s; jitter x0.99-1.01, level -20 +-1 dBFS, random phase,
seed 20261106. Paste layers {14, 16, 18, 20}, audio
positions, donor span checked (fixpack machinery). Probe
reads at the paste site (tuple entry layer+1; sanity: must
read donor) and every downstream state to 36, at audio-mean
and final positions. The same patched forwards report the
pitch lean (premise check: stays at the recipient's).

**Stimuli caveat.** Probe-cell tones are generated on
digital silence (no background scheme), seed-fixed; the
probe trains and is tested on the same recipe, so the cell
is internally valid. Absolute validity MAE is compared to
the graded battery's 0.05-0.09 octaves with that caveat.

**Registered predictions.** (1) Unpatched validity MAE
<= ~0.10 octaves at audio positions, states 5-36.
(2) Paste site reads the donor (error < 0.1 octaves).
(3) The cell: downstream predictions closer to the DONOR
for most pairs/layers (fine pitch is in the state; the lean
ignores it) OR staying at the recipient (the state itself
is band-coarse mid-window). Either outcome is a result.

**Builder.** `colab/build_probecell_qwen_notebook.py`
(fixpack runner head + machinery whole, battery replaced;
own imports in the appended parts; save via the head's
save_both; traceback printed into the log on any cell
error). Notebook `project_k_qwen_probecell_colab.ipynb`,
11 cells. Outputs: probecell_probe.csv, probecell_lean.csv,
probecell_run_log.txt in the tracing-qwen3b Drive folder.
