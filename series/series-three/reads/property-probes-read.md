# Step 5 read: property probes on Qwen2.5-Omni-3B (Nov 3, 2026)

The Handover Correction, step 5. Question: round 2 showed a checker can
read sound-vs-silence at every layer, but presence is a coarse question
and a flat perfect curve cannot localize a subtler loss. So this battery
asks about two properties of the sound itself, in the layers the
surgery found interesting: pitch (500 vs 1500 Hz) and onset (burst at
0.5 vs 2.5 s), factorial, 90 instances per cell, 360 tones, plus 60
silence clips. Splits are by instance within each cell; silence
waveforms are grouped by hash so no silence stimulus sits on both
sides of the split. Convention: state k is the output of decoder
layer k-1; the focus band, states 16 to 25, is decoder layers 15 to 24.
Source: property_probe_results.csv (370 rows), property_manifest.csv,
property_run_log.txt, all in this folder and on Drive. Run 4 completed
clean Nov 3, 2026 (token check passed, 420 of 420 extracted, exit 0).

## The numbers

Pitch, high vs low, read at the audio positions (mean-pooled):
1.000 test accuracy at every state, 0 through 36, both readouts
(plain and PCA-64). Pitch is in the audio embeddings from the start
and never leaves.

Pitch at the final answer position: 0.500 at state 0 (the embedding
output, where that position is still just the last prompt token),
then 1.000 at every state from 1 to 36, both readouts.

Onset, late vs early, at the audio positions: 0.483 (PCA-64) and
0.525 (plain) at state 0, then 1.000 at every state from 1 to 36.
The state-0 miss is expected and reassuring: before any decoder
layer runs, a mean over the audio positions cannot tell an early
burst from a late one, because the same burst sits in the average
either way. One layer of attention puts position information into
the pooled readout and the property becomes perfectly readable.

Onset at the final answer position: 0.500 at state 0, then 1.000 at
every state from 1 to 36.

Controls: 200 label permutations per focus-band state for the two
property tasks (PCA-64 readouts, both positions). Shuffled means sit
at 0.486 to 0.515, i.e. chance; the 95th percentile of the shuffled
distribution tops out at 0.718 (onset, audio positions). The real
readers score 1.000 at every focus-band state, far above the null.
The null spread is wide (std up to 0.12) because all 200 permutations
are scored on the same 120-clip test set; the gap between 1.000 and
0.718 is still large.

Presence sanity task (tone vs silence, audio positions): flat
0.9375 at every state. This number is uninformative as run: the
test split holds 120 tones and only 8 silence clips (the grouped
split put the large digital-silence group in train), so 0.9375 is
exactly the always-answer-tone score, and the shuffled controls for
this task also score 0.92 to 0.96. The presence claim rests on
round 2's balanced battery, not on this one. Recorded here so the
flat line is never quoted as a presence result.

## What this settles, and what it does not

Settled: in Qwen 3B, two properties of a tone, its pitch and when it
starts, stay linearly readable at every stage of the network,
including at the final answer position from state 1 onward, across
and beyond the band where the surgery found the presence decision
stops being movable by audio-slot patches (after decoder layer 18)
and where the relay hands the decision to the question positions and
then the final position. The information the model's answers fail to
use is present, in readable form, at the place the answer is
written, at the time it is written.

Not settled: decodability is not use. These probes show the
properties can be read by a trained checker; they do not show the
model's own answer machinery reads them, and the behavioral record
(open-form naming fails, timing answers collapse early) says it
mostly does not. The causal version of this test is the property
patch: donors matched on one property and mismatched on the other,
patched at layers 15 to 24, scored on whether the answer's property
choices move. That is the follow-up the council suggested; it has
not run.

Scope: one open model, synthetic tones at one loudness band, one
sentence-free stimulus family. Nothing here is a claim about other
models or about natural sounds.

## The Handover Correction, closed

Step 5 was the last open step. The correction's five steps are all
done: the wrong-positions duplication verified against the data and
the handover claim withdrawn; the control rerun as an independent
construction, which found the staged relay (audio slots to layer 18,
question positions strongest at 19 to 22, the final position alone
carrying the decision by layer 28); the layer-19 boundary re-reported
under both readouts (a fade zone, 19 to 27, readout-dependent); the
codec battery rebuilt like-with-like with full stimulus provenance;
MiniCPM held until these probes landed. It has now landed. MiniCPM
work is unblocked by the correction and waits only on sequencing.
