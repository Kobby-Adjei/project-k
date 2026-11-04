# Surgery read: activation patching, Qwen 3B (Series Three, test 2)

Runs Nov 3, 2026, free lane, L4. Token check passed (audio token 151646, 100
audio positions through this notebook's audio-only feed), feed proof passed on
all 20 clips, sweep completed: 744 of 744 rows. The battery's final stage
(best-layer open-form check) failed on a row-length guard in the first run and
was rerun clean the same evening with a fix; its result is in the scorecard.

Baselines (unpatched presence scores): tones 0.80 to 0.89, all Yes. Silences 0.3257,
all No. Speech 0.9107, Yes, under the speaking question. Open-form baselines:
tone_00 answered "silent" (the miss this test exists to explain), silence_00
"silent", speech_00 "speech synthesizer".

## The sweep

Tone into silence (8 pairs): the yes score rises from the silence baseline 0.33 to
0.86 when the patch lands at layer 0, holds a high plateau through layer 18
(0.69 to 0.86), then collapses to about 0.33 from layer 19 to 35. The rescue works
for the first half of the model and stops at layer 19.

Silence into tone, the reverse control (4 pairs): the mirror image. Patching early
(layers 0 to 18) drags the tone run down to 0.32 to 0.49; patching late (19 to 35)
changes nothing, 0.78 to 0.86. The crossover sits at the same layer.

Tone into tone, the sham control (4 pairs): flat, 0.85 to 0.87 at every layer.
Patching a tone into a tone run changes nothing, as it should.

Wrong positions, the control that was supposed to do nothing (4 pairs; only the
non-audio positions patched, at layers 0, 7, 14, 21, 28, 35): 0.32, 0.35, 0.41,
0.84, 0.85, 0.50. No rescue early, full rescue at layers 21 and 28, partial at 35.
WITHDRAWN Nov 3, 2026 (The Handover Correction): this control is not an
independent measurement. Its rows at layers 0, 7, 14, 21, 28 are numerically
identical to the mirrored silence_to_tone rows in both sweeps (20 of 24 rows,
score and answer word), because the two constructions assemble the same composite
state (silence audio states plus tone non-audio states). It therefore corroborates
nothing on its own. The claim that the deciding content sits in the non-audio
positions at layers 21 to 28 was withdrawn, and the independently constructed
rerun has now landed: see handover-rerun-read.md. The corrected picture is a
staged relay (audio slots to layer 18; question positions strongest 19 to 22;
final answer position by 28), not the single jump this control suggested.

Speech into silence (4 pairs, one donor sentence in four copies): the same
crossover. High through layer 18 (0.77 to 0.91), collapse at 19, falling to about
0.14 in the late layers, below the silence baseline. Noted as observed; the
below-baseline tail is unexplained and the arm is effectively one donor.

## What it means

What the untainted arms support: patching the audio slots changes the presence
decision at layers 0 to 18 and stops doing so from layer 19, in both directions
(tone into silence rescues; silence into tone kills). The stronger claim, that
the content hands over to the wider stream and stays decisive there, is
WITHDRAWN (see the wrong-positions note above) pending the independent rerun.
Nothing here shows the sound being erased, and round 2's checkers found its
trace readable to layer 36, subject to that round's own caveats. The open-form
"silent" for a loud tone cannot be blamed on the sound dying inside the model.

## Predictions scorecard

- "Patching rescues the tone yes/no at mid layers": confirmed in spirit, wider and
  earlier than predicted (layers 0 to 18), with a sharp boundary at 19 in the
  yes-score readout. (The emitted-answer readout is ragged and ends nearer 27;
  the dual-readout re-report is part of The Handover Correction.)
- "Patching does not fix the speech denial": the premise did not occur here. With
  sound alone and no picture, there is no denial to fix; speech baselines are Yes
  and the open form names it. The denial belongs to the picture conditions.
- "Patching does not rescue the open form": mostly confirmed, with a real
  exception. The stage reran clean Nov 3, 2026 (fix: donor captures taken under
  the open question itself). At the best layer (layer 0, mean lift +0.5377), the
  8 patched silence runs answered the open question: 6 recorded "silent" and
  2 (donors tone_05 and tone_07) recorded "beep" (patching_openform.csv, the
  record of this stage). Presence rescue is total; open-form rescue is
  rare and donor-dependent. Naming is a later, harder step than detection.
  Correction (Nov 4, 2026): an earlier version of this read quoted the two
  answers as full sentences ("I cannot hear anything in this audio." /
  "A continuous high-pitched beeping sound."). Those sentences came from a
  run report's narration, not from the data file; the file records the
  one-word answers above. Claude's Nov 4 audit caught the mismatch. The
  finding is unchanged (6 of 8 still deny, 2 of 8 name the beep); the
  quotes are now the file's.

## Caveats

The transplant replaces all 100 audio positions at once; it is a whole-sound
transplant, not a tone-window-only patch. The layer 19 boundary marks where the
content stops being carried in the audio slots, not where all trace of the sound
ends. Synthetic clips, one model, means over 8 or 4 pairs.

## The Handover Correction, step 3: the boundary under both readouts

Layer convention, fixed here for all Series Three reporting: "layer L"
means the output of decoder layer L (36 decoder layers, 0 to 35). The
probe table's states are indexed 0 to 36, where state L+1 is decoder
layer L's output and state 0 is the embedding output; probe accuracy
held at every state.

The two readouts disagree in 80 of the 744 rows (a score of 0.5 or more
with a No word, or the reverse), so the boundary is reported under both.
Tone into silence, means over 8 pairs, layers 14 to 30:

- By yes score: the plateau holds through 18 (0.7126), collapses at 19
  (0.3686), and sits at the silence baseline from 20 on.
- By emitted answer: 8 of 8 Yes at 19, again at 22 and 23, ragged between
  (2 of 8 at 20, 3 of 8 at 21, 7 of 8 at 24, 4 of 8 at 25, 2 of 8 at 26),
  and 0 of 8 only from 27.

So the sharp cliff at 19 is a property of the score readout. Under the
emitted-answer readout the audio slots' influence fades raggedly across
layers 19 to 26 and ends at 27. The honest statement is a fade zone,
19 to 27, whose shape depends on the readout, over an early region
(0 to 18) where both readouts agree the slots are decisive. The reverse
arm shows the same split in mirror image (its score is dragged to
0.40 to 0.49 at layers 14 to 18 while its emitted word stays Yes).


CORRECTION (Nov 4, 2026, after the index check; see
indexcheck-read.md): every layer-35 value in this file is a
double-norm artifact. In the pinned transformers stack, the
hidden-state tuple's last entry (index 36) is the final state
AFTER the model's closing norm, not decoder layer 35's raw
output, so "layer 35" patches pasted an already-normed vector
into a slot that gets normed again. Layers 0-34 were patched
correctly and stand as labeled. Read any layer-35 row above as
an artifact, not a measurement.

WITHDRAWN Nov 4, 2026 (Claude's follow-up audit, verified against
the files): the scorecard line above, "Presence rescue is total;
open-form rescue is rare and donor-dependent. Naming is a later,
harder step than detection," and any "paste fixes noticing more
than naming" framing. The index-check battery ran all 8 tones
unpatched under the open question (openform_tones_unpatched.csv):
6 "silent", 2 "beep" (tone_05, tone_07), identical clip-for-clip
to the patched answers in patching_openform.csv. The layer-0
paste reproduced each donor's own answer exactly, naming
included, so those cells measured a wholesale transplant, not a
noticing-versus-naming gap. The informative open-form cells
would be the partial-rescue layers (14-24); they have not run.
