# The fixpack: build note (Nov 4, 2026)

**What it is.** One notebook carrying the council's Nov 4 corrections to the
property patch, in their order: (1) gap-matched sham reconstruction, (2) the
word_matches_donor fix, (3) open-form controls, (4) the four-wording readout,
(5) the mediation cell. Built at Kobby's instruction on the Nov 4 voice call:
"Make sure is the corrections that they gave you."

**Builder:** `colab/build_fixpack_qwen_notebook.py` (a copy of the proppatch
builder with the battery block replaced). Notebook:
`project_k_qwen_fixpack_colab.ipynb`, 12 cells. Runner head = the property
runner's head with outputs renamed fixpack_, plus three fixes baked in:
- `rms_db_np` defined in the head just before `feed_inputs` (Claude's
  residual: the borrowed head's broken-feed guard called it; its definition
  lived in the cut tail).
- `patched_share` now computes the DONOR's audio block too and hard-stops
  (exit 12) if it differs from the recipient's (ChatGPT's residual; positions
  were previously taken from the recipient only).
- The appended parts carry their own imports (the proppatch attempt-2 lesson).

**Parts.**
- A: gap-matched shams (12 pairs): the donor matches the recipient on the
  target property and differs on the other property by a full gap. Verified
  against the behavior file before the run: sham target gaps average
  0.041 (pitch arm) / 0.016 (onset arm); sham other-property gaps average
  0.131 / 0.116, the size of the real gaps (0.103 / 0.126). Sham recipients
  start at index 3 of each cell: zero pair overlap with the real pairs and
  between the sham arms (checked locally). Layers
  {0,14,16,18,19,20,21,22,24,28,34}, both questions read. Plus the 24 real
  pairs at layer 21 only (the grid layer run 1 lacked; needed beside the
  mediation cells). word_matches_donor compares to the donor's OWN
  unpatched word; the donor's word is now a CSV column.
- B: the 12 directional pairs at layers {0,19,22}: the donor's audio state
  is captured ONCE under the pitch question (audio positions precede the
  question tokens, so under causal attention the capture cannot depend on
  the wording) and read under all six wordings (2 originals, 2 reversed,
  2 choices, byte-identical to the closers battery). Output:
  fixpack_wording.csv.
- C: mediation on the same 12 pairs at layers {19,20,21,22}: donor state at
  the audio positions with the recipient's own state restored at the
  post-block positions. Rows in fixpack_results.csv with mode=mediation.
- D: open-form controls on the 6 pitch low-to-high pairs: recipient
  unpatched, donor unpatched, real pastes at {28,34} (14/19/24 exist from
  run 1), gap-matched sham pastes at {14,19,24,28,34}. Output:
  fixpack_openform.csv with a kind column.

**Registered predictions (written before the run).**
- A: if the paste is wholesale, the sham's OTHER-property readout moves by
  ~100% of its (full) gap and its target readout moves by ~100% of its
  small gap too; if any property-specificity exists at the audio slots,
  the target readout's pct will sit well below the other readout's.
- B: the paste will move the originals' and choices' margins; whether it
  moves the reversed pitch margin (an onset read in the wording battery)
  is the open question. No prediction registered beyond that.
- C: if the post-block positions carry the effect at 19-22, the mediation
  cells fall back toward the recipient's own lean; if the effect survives
  mediation, the carrier is elsewhere (or the restore is too late).
- D: if "beep" appears for unpatched recipients, the 18/18 was the
  battery, not the paste.

**Local checks before the run.** Runner compiles (48,267 chars at first
build). Static name audit: no undefined names beyond the known `_e`/`e`
false positives. Pair plan executed against the local manifest: counts and
disjointness as above. The fuller post-position grid (all ten layers, both
directions) from Claude's (a) discussion is NOT in this run; it was not in
either reviewer's stated order. Deferred, not dropped.
