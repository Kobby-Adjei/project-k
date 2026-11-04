# Handover Correction, step 2 read: the independent rerun (Nov 3, 2026)

Run Nov 3, 2026, free lane, L4. Token check passed (100 audio positions),
feed proof passed on all 20 clips, 180 of 180 runs, exit 0. Baselines match
the original battery exactly (tone_00 0.8811 Yes, silence_00 0.3257 No).
Data: handover_rerun_results.csv. Design: handover-rerun-build-note.md.
Independence check: exactly 1 of 180 rows shares a yes score with a mirrored
reverse-control row, a no-effect cell at the shared baseline value (0.3250);
the original cell shared 20 of 24. These rows are new measurements.

## The three subsets (means over 4 pairs; yes words out of 4)

Pre-sound text positions: nothing, anywhere. Mean yes score 0.323 to 0.327
at all 15 layers, 0 of 4 Yes words at every layer. The text before the sound
cannot carry it (it has not seen the sound yet). This arm is the rerun's own
negative control, and it is flat.

Question positions after the sound, final position excluded: the emitted
word flips to Yes from layer 14, and the score climbs to a peak in the
19 to 22 window (0.7720, 0.7987, 0.7872, 0.7796), then falls (0.7281 at 23,
0.5993 at 24) and is gone at 28 (0.3453, 0 of 4). The question region holds
the deciding content, in a window centered just after the layer 19 boundary.

Final answer position alone: nothing through layer 22 (0.34 to 0.37, 0 of
4), the word flips at 23 (0.4605, 4 of 4), rises at 24 (0.6122), and fully
rescues at 28 (0.8372, 4 of 4). Partial at 35 (0.4988, 3 of 4). By layer 28
the whole decision can be carried by the single position that predicts the
first answer token.

## The corrected claim

Withdrawn claim (Nov 3, 2026, earlier today): by layers 21 to 28 the
deciding content sits in the non-audio positions. It rested on a control
that was not independent, and its detail was also wrong: at 28 the question
region no longer rescues at all.

Corrected claim, on independent constructions: the tone's deciding content
changes address in stages. It is carried in the audio slots through layer
18 (the original sweep, both directions). It is then rescuable from the
question positions after the sound, strongest at layers 19 to 22. By layer
28 it has consolidated into the final answer position, which alone
transfers the decision. A handover happens, but it is a relay with named
stops, not a single jump, and the stops are at different layers for the
question region (peak 19 to 22) and the final position (28).

Caveats: 4 pairs, one model, synthetic tones; the post arm's word flips at
scores of 0.35 to 0.45 in layers 14 to 18 (read both readouts; see the
dual-readout re-report in patching-read.md); the layer 35 final-position
value (0.4988) sits downstream of almost nothing and is noted, not read.


CORRECTION (Nov 4, 2026, after the index check; see
indexcheck-read.md): the final@35 cells above ("Partial at 35
(0.4988, 3 of 4)") are WITHDRAWN. The index check showed the
break is at the last index only: in the pinned transformers
stack, hidden-state index 36 is the post-norm final state, not
decoder layer 35's raw output, so those cells pasted an
already-normed vector into a slot that gets normed again.
The staged-relay reading otherwise stands on layers 0-28,
which the index check verified exact.
