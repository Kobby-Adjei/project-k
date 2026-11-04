# The property patch, read (Nov 4, 2026)

The causal step in the council's order. 792 rows in
proppatch_results.csv (396 patch cells, two questions each),
read from the file here. The battery's open-form cells failed
in the main run (the generation call; see the attempt log in
proppatch-build-note.md) and landed in a fixed rerun — see
the open-form section below. Everything below is the margin
and word data, which is complete.

## Design recap

Recipients and donors are independent property tones, matched
on the other property, opposed on the patched one (6 pairs per
direction). The donor's states at the audio positions are
pasted into the recipient's run at one decoder layer; the
readout is the recipient's yes share (the lean) and first word
under the original pitch and onset questions. Layers {0, 14,
16, 18, 19, 20, 22, 24, 28, 34}. Controls: sham pastes
(same-value donor) and post-position pastes at {0, 19, 28}.
Each cell also reads the other property's margin.

## The result

Pasting at the audio positions moves the recipient's lean to
the donor's value, in a window that closes in the early 20s:

- Pitch, low recipient with high donor: the lean moves toward
  the donor by 102% of the mean donor gap at layer 0, 91% at
  18, 53% at 20, 30% at 22, and nothing from 24 on (0.006,
  0.001, 0.003 at 24, 28, 34). The reverse direction mirrors
  it: 101%, 93%, 51%, 21%, then nothing.
- Onset, both directions: 100-116% of the gap at layers 0-20,
  15-21% at 22, nothing from 24 on.
- The spoken word never moves: across all 396 onset cells the
  first word is Yes 396 times; across the pitch cells, Yes 393
  of 396. A full-strength transplant of the property at the
  sound's slots changes the lean completely and the answer not
  at all, because both classes' leans live on the yes side of
  the fixed cut. This is the threshold account confirmed
  causally: the paste moves exactly the quantity the audit
  said was perfectly ordered, and the quantity that produces
  answers does not budge.

## The controls

- Sham pastes (donor with the same property value, different
  clip): moves of 0.000-0.022, against 0.08-0.15 for the real
  pastes. The effect is the property, not the pasting.
- Post-position pastes: nothing at layers 0 and 28; a small
  real leak at 19 (pitch +0.036, onset +0.019, roughly a fifth
  to a third of the audio-position move at that layer). By
  layer 19 some property influence already sits past the
  sound's slots, consistent with the handover rerun's
  post-block window. It is a leak, not the channel: the audio
  positions carry the full effect at the same layer.
- Specificity: a pitch paste moves the onset lean by at most
  0.032 (layer 0) and 0.02 or less elsewhere; an onset paste
  moves the pitch lean by at most 0.026. The two properties
  travel separately.

## What stands

- The property is causally set at the audio positions through
  about layer 20: replace those states and the model's lean
  becomes the donor's, for pitch and for onset, in both
  directions, with property-specific effects and clean
  controls. From layer 24 the same paste does nothing: by
  then the property's influence has left the sound's slots.
- Combined with the closers: the lean that the probes read at
  every layer is not a passive trace. In the early window it
  is the working copy the answer is computed from; the failure
  to answer correctly sits downstream, at the yes/no cut,
  which no audio-position paste at any layer can move because
  the cut itself is what is misplaced.
- Prediction scorecard (registered in the build note): the
  band exists and the sham stays near zero, as predicted; the
  specificity prediction held; the post-position prediction
  ("does not move") was wrong at layer 19 by a small amount,
  recorded above.

## The open-form cells (landed Nov 4, 2026, after two fixes)

The pitch arm's six low-to-high pairs, pasted at layers 14,
19, and 24, asked the open question: all 18 answers are
"beep" (proppatch_openform.csv, verified from the file). The
recipients are low tones whose naming is the weak direction
in this model; under a high donor's states at the sound's
slots, every one of them names the sound, at all three
partial layers, including 24, where the margin paste has
already faded (the open answer and the margin need not share
one window, and layer 24's post-block leak is a candidate
carrier).

One scope note, stated plainly: this battery did not record
the same clips' unpatched open answers, so the comparison
rests on the neighboring record (in the index check, the 8
unpatched patching tones named the sound in 2 of 8; in the
codec battery the loud reference tone answered "silent" in
open form through both arms). A six-forward unpatched control
on these exact clips would close it, and is the cheapest open
item in the lane.

## Open

- The unpatched open-form control for these six clips (above).
- Next in the council's order: MiniCPM.
