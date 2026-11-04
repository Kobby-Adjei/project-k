# Series Three: the inside-tests protocol (frozen Nov 4, 2026)

The battery as a recipe, written so it can be run on a
second model without redesign. Only the corrected version
of each cell is canonical. Designs the audits voided are
listed at the end so they are never rebuilt by accident.
Predictions are registered in writing before each run;
results are read from the output files, never from run
narration.

## The adapter: prove the kitchen before any number

A new model's stack is guilty until proven innocent.

1. Index convention. Derive, from the model's own
   library source, what each hidden-state tuple entry
   is. Then prove it empirically: all-positions pastes
   must reproduce the donor exactly at three widely
   spaced layers, and a paste of the final tuple entry
   into the post-norm slot at the final position must
   return the donor baseline exactly. No patch number
   is reported before both pass. (On Qwen's stack the
   last entry was the post-norm state, and every
   "layer 35" patch was a double-norm artifact.)
2. Sound positions. Take them from the model's own
   machinery (token histogram, bounds, or splice
   points), per clip, with hard stops for empty or
   non-contiguous spans. For every paste, evidence
   that donor and recipient spans are the same list;
   comparing total lengths is not evidence.
3. Load. Pin the library version the model requires;
   precision never drops to fit a smaller GPU; the
   load ladder and every fallback are logged.

## Stimuli

Synthetic tones with exact ground truth: factorial pitch
x onset instances (unique waveforms, jittered pitch,
onset, level, phase), silence controls de-duplicated by
waveform hash. Splits are by waveform identity, never by
clip name alone. File-level provenance (hashes, recipe,
bitrates) is logged for every stimulus set, and any
behavioral comparison holds the input door constant —
a codec contrast and a door contrast are never mixed in
one arm.

## The battery, in canonical order

1. Codec control. The same sound raw and after the
   project's compression recipe, through ONE door;
   then the door contrast separately. Open answers and
   margins both read.
2. Probes. Per-layer linear readouts at the sound's
   positions and the final position, regularized, with
   a principal-component readout beside the plain one;
   200-permutation nulls reported as mean and 95th
   percentile. Graded targets (exact frequency, exact
   onset) beside binary ones, so a ceiling cannot hide
   decay. A probe result is decodability, never use.
3. Behavioral questions. The model is asked the facts
   directly (yes/no in both directions, forced choices),
   scored by word, by margin at the fixed cut, and by
   AUC. Margin claims name their wording.
4. Presence surgery. Donor states pasted at the sound's
   positions, one layer at a time, into silence
   recipients and back; sham pastes; strict-subset
   position controls that cannot assemble the hybrid
   state. Both readouts (margin and word) reported
   side by side; disagreements counted, not hidden.
5. Property patch. Independent donors matched on the
   other property, opposed on the patched one. Sham
   donors are GAP-MATCHED: same value on the target
   property, a full gap on the other. The word column
   compares against the donor's OWN unpatched word,
   never its label. Open-form cells sit at partial
   layers, in the same load as their controls,
   including a donor matched on BOTH properties — the
   only true no-effect control for naming.
6. Mediation. Restore the recipient's post-block rows
   at EVERY layer from the paste layer up, the final
   position excluded and the restored set named. Four
   arms: paste alone, paste + restore, restore alone
   (must return the baseline bit-for-bit — the hook
   validation), sham paste + restore.
7. Probe cell (when a within-band question arises).
   Paste donors whose target-property difference the
   margin cannot see, and read the patched states with
   the graded probe: does the fine value travel in the
   state where the margin is blind to it?

## Reporting rules (learned, binding)

- Effects in logit steps of the margin, with percents
  of own-pair gap as a secondary view. Averages over
  tiny gaps are not point values.
- Baselines for every clip before any paste; the cut's
  location is measured and AUC reported, never only a
  fixed 0.5.
- Every real/sham contrast in one notebook and one
  model load.
- The code ships with the results. A claim an outsider
  cannot recompute from the shipped files is not a
  frozen claim.
- Corrections are appended in place with dates, in the
  reads and in the journey file; superseded numbers are
  never silently replaced.

## Void designs (do not rebuild)

- Sham donors drawn from the recipient's own cell
  (gaps 5-8x smaller than the real arms; normalized,
  the sham curve is the real curve).
- Wrong-position controls assembled by reversing a
  paste (they rebuild the same hybrid state and
  duplicate another arm's rows).
- Single-layer post-block restoration (restores rows
  the paste never changed; tests nothing).
- Word columns scored against manifest labels.
- Two-arm contrasts split across notebooks or loads.
