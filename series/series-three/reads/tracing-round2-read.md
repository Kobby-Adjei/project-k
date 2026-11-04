Series Three, test 2 (tracing), round 2. Series named Nov 3, 2026.

# Tracing round 2 read (Qwen2.5-Omni-3B) - Nov 3, 2026

Run: completed clean Nov 3, 2026 (battery exit 0). 1,080 clips (360 tone / 360 speech /
360 silence, 4 s each, sound varied in pitch, timing, loudness). Probes trained per
layer 0-36 on pooled audio-position features, 480 train / 240 test per task, two
readouts (plain, PCA-64), controls = mean of 5 label shuffles per layer.

Provenance note: the first attempt was interrupted by a stray click AFTER extraction
finished (correcting the earlier "extraction interrupted" phrasing). The rerun found
the complete 1,080-clip features file, verified the count against the manifest (the
notebook stops on any mismatch), reused it, and trained fresh probes. Data chain sound.

## Probe results: flat at the ceiling

Accuracy 1.00 at EVERY layer, 0 through 36, for tone_vs_silence and speech_vs_silence,
in both readouts. Shuffled controls: means 0.36 to 0.64 (chance is 0.50), no layer
anywhere near 1.00. The control rule passes this time, with one note: the shuffle
spread is wide (std 0.15 to 0.21), so individual shuffles wander; the means sit at
chance and the gap to 1.00 is total.

What this says: the presence of the sound stays readable in the model's own numbers
all the way to the final layer, for the tone and for the speech. No layer erases it.
That kills the "lost in the middle" story for presence, and it matches the speech
prediction exactly (readable everywhere; the denial lives in answer formation).

What it does NOT say, the saturation caveat: presence is a coarse question. The
transformer stream carries earlier material forward by addition, so a leftover trace
of "something was in the audio" can ride to the end without the tone being held
anywhere as a usable fact. A flat 1.00 curve cannot localize a loss because the
instrument topped out. The sharper probe asks for properties a trace cannot fake:
which pitch (several tones), or when the sound started.

## Attention results: the answer does not look at the sound

Attention share to the audio positions (12-clip sample, tone and silence, layers
0-35): mean about 0.001 for tone and about 0.001 for silence, maximum 0.0029 anywhere.
The answering side attends to the sound positions the same whether a tone is playing
or nothing is, and barely at all in absolute terms. Consistent with "present but not
used." Attention is the weak instrument (a hint, not a proof); surgery is still owed.

## Predictions scorecard (from tracing-plan.md, written before the runs)

- Tone readable into the mid layers: WRONG in the interesting direction; readable to
  layer 36.
- Speech readable at every layer: CONFIRMED.
- Patching rescues the tone yes/no; patching does not fix the speech denial: UNTESTED.
  The surgery step is now the load-bearing test of the whole tracing design.

## Verdict

Round 2 is valid where round 1 was void: right token (250 <|AUDIO|> positions),
enough examples to force reading over memorizing, controls at chance. Its answer is
coarse but real: in Qwen 3B, neither the tone nor the speech is erased inside the
model. Presence survives to the end; the answers ignore it. The loss sits past
readability, in use, naming, or answer formation. Next discriminating steps:
(1) the surgery (activation patching) from the plan, (2) a property probe (which
pitch / when) if a finer map of what survives is wanted.
