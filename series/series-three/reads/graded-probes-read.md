# The graded probes, read (Nov 4, 2026)

Run: Nov 4, 2026, free lane (kadjei1), L4, exit 0, token check
passed. Files: graded_probe_results.csv (592 rows),
graded_manifest.csv (360 tones, split 240 train / 120 test),
graded_run_log.txt, in the Drive tracing-qwen3b folder with local
copies beside this file. All numbers below are read from the CSV.

Why this battery exists: step 5's binary checkers (500 vs
1500 Hz, 0.5 vs 2.5 s) scored 1.000 at every state, and the Nov 4
council audit pointed out that a ceiling score cannot show decay.
These probes regress the actual values instead: 360 tones, 30
pitches log-spaced 300-2400 Hz crossed with 12 onsets 0.25-3.25 s,
each jittered, the jittered values being the targets. Ridge
readouts (PCA-64 headline, plain ridge as the second readout) at
the audio positions (mean-pooled) and the final position, every
state 0-36, on two splits (by clip; held-out pitch values), with
200-permutation nulls at states 16-25.

## The curves (by-clip split, PCA-64 readout)

Pitch, audio positions: R2 0.992 at state 0, 0.991 at 19, 0.980
at 36. Error (MAE, octaves): 0.052-0.056 through state 10,
drifting up to 0.094 at state 36. In frequency terms that is
about 4% error early, about 7% at the last state.

Pitch, final position: R2 -0.000 at state 0 (the built-in sanity
line: no layer has run, the final position has not seen the
sound; onset there is also chance, R2 -0.037). By state 5 the
final position reads pitch at R2 0.980 (PCA-64 readout, by-clip
split), and it ends at 0.983
(MAE 0.086 octaves), slightly better than the audio positions.

Onset, audio positions: weak at state 0 (R2 0.277, MAE 0.68 s;
an average over positions cannot localize before any layer has
mixed them), then R2 0.947 at state 1 and 0.99+ from state 10
(with one exception, state 33 at 0.9897),
MAE about 0.05 s, flat to state 36 (0.053 s).

Onset, final position: chance at state 0, 0.985 by state 5,
MAE 0.049-0.074 s across the back half, ending 0.074 s.

Controls: the held-out-pitch split (train on half the pitch
values, test on the other half) matches the by-clip numbers
closely everywhere (pitch at audio, state 36: MAE 0.097 vs
0.094), so the readouts interpolate; they are not memorizing
the grid. The plain readout traces the same curves with
slightly smaller errors in several cells. Permutation nulls at
states 16-25 sit at R2 mean -0.14 to -0.25, 95th percentile at
most 0.1343, against real R2 of 0.976-0.994. The nulls cover
only the focus slice (40 of 592 rows: states 16-25, PCA-64,
by-clip); the margin in that slice is the claim.

## Verdict against the registered predictions

Predictions written before the run: pitch MAE under ~0.06
octaves and onset MAE under ~0.2 s at every state; final
position at state 0 at chance; decay, if any, as error rising
with state.

- Onset: held, everywhere from state 1 at both positions.
- Final position at state 0 at chance: held, both tasks.
- Pitch: the number fails late. Audio-position pitch error
  crosses 0.06 octaves around state 16 and reaches 0.094 at
  state 36; the final position sits at 0.084-0.102 through the
  back half. There IS a mild upward drift with depth, exactly
  the direction the audit said a ceiling score would hide.
- But there is no decay in any stronger sense: R2 never falls
  below 0.976 for pitch or 0.985 for onset (after state 5) at
  either position, and the worst pitch error anywhere is a
  tenth of an octave. The graded information is essentially
  intact at every state, with a small late drift in pitch
  precision at the audio positions.

## What this does to the Series Three picture

Paired with the same morning's behavioral battery: a ridge
readout at the model's own final position recovers pitch to
about a tenth of an octave and onset to about a fifteenth of
the clip, at every late state, on stimuli of the same family
the model answers about at the base rate (Yes on 360 of 360
onset asks). CORRECTED Nov 4, 2026 after Claude's follow-up
audit: the pairing is sharper than a use gap. The model's own
yes/no margin on those asks separates pitch and onset almost
perfectly (AUC 1.00 and 0.999, recomputed from the behavioral
file); the wrong answers come from the fixed 0.5 threshold
sitting below the whole margin distribution. What is measured
in graded units now is the precision of what arrives at the
answer position, and the behavioral failure is located at the
yes/no cut, not in transmission. The remaining causal step in
the council's order is the property patch; MiniCPM after.
