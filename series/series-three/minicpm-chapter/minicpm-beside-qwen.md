# MiniCPM beside Qwen: the Series Three comparison

Written Nov 10, 2026, when the MiniCPM chapter of Series
Three was completed (cells 1 to 7, Nov 7 to 10, 2026).
Qwen 3B's chapter was frozen on Nov 4, 2026; its numbers
below are the frozen ones, from the frozen reads. Every
MiniCPM number was recomputed from its run's CSV files
during this chapter; the per-cell reads in this folder
are the record. This page is the MiniCPM addendum to
the chapter record. The frozen Qwen evidence ledger is
not edited by it.

The two models were run through the same frozen
protocol, in the same order, with the same stimuli
wherever the protocol fixes them (the factorial tones
are hash-identical in both chapters). The feeds differ
by necessity: each model was asked through its own
chat-shaped door, and each paste used that stack's own
proven convention (Qwen's state index against MiniCPM's
entry, the state entering a decoder layer). Depths are
compared as shares of the stack: Qwen has 36 layers,
MiniCPM 28.

## The short version

The two chapters tell the same story with the same
shape, and they differ in three places that matter:

1. Both models keep the sound. In both, the file is not
   the loss, the properties are readable from the
   inside states, pastes transplant them inside a
   window that closes partway through the stack, the
   effect runs through the post-sound positions, and
   the spoken answer fails at the reporting step, not
   the hearing step.
2. MiniCPM's margins are noisier and its onset is
   fainter. Qwen's pitch ranking was perfect (AUC
   1.0000) and MiniCPM's is strong but imperfect
   (0.9476). Qwen's onset contrasts could be separated
   by paste; MiniCPM's onset scores sit so close
   together between instances that the specificity test
   cannot separate a real onset paste from a sham.
3. The depths line up. Qwen's presence crossover sits
   at about 53 percent of its stack; MiniCPM's closes
   at entries 15 to 16 of 28, about 54 percent. The
   paste-alone fade in mediation lands on the same
   crossover in both chapters' terms.

## Cell by cell

**Codec control (cell 1).** Both: the compression is
not the loss. Qwen's two arms agreed on the battery;
MiniCPM's agreed on all 28 asks (Nov 7, 2026). In both
chapters the speech sound was denied in the report
while silence drew errors of its own kind (Qwen's
silence behavior differed in detail; MiniCPM answered
a false Yes to beeping on an empty track).

**Probes (cell 2).** Both: the clip-level split reads
perfectly (1.0000 at every layer in both), and the
honest waveform-grouped split exposes the same silence
leak in both (tone against silence at the audio
positions, 0.6704 at every layer, identical numbers,
confirmed by the files-only Qwen recheck on Nov 7,
2026). Speech separates in both, earlier in Qwen,
from mid-depth in MiniCPM.

**Behavioral questions (cell 3).** Both: the spoken
word follows the form of the question and the margin
underneath ranks the truth. Qwen's margins were
perfect or near (pitch AUC 1.0000, onset 0.9988; best
cuts 360 of 360 and 357 of 360). MiniCPM's rank a
little lower (0.9476 and 0.9831; best cuts 324 and 341
of 360) and its defaults differ by form (yes on the
pitch form, no on the onset form, no on both reversed
forms), where Qwen leaned to the affirmative and the
first-named option. Both take the first-named option
in forced choice almost always.

**Presence surgery (cell 4).** Both: the window
closes in the second half of the stack (Qwen layer 19
of 36; MiniCPM entries 15 to 16 of 28, the same
relative depth), speech states do not carry the beeps
answer, and the strict-subset controls show the same
overlap: the post-sound positions carry the effect
first, the final position carries it later, and the
two overlap in between. MiniCPM's sham was flat in
absolute terms, as Qwen's was.

**Property patch and naming (cell 5).** Both: pitch
pastes are strong inside the window and specific
against the gap-matched sham (Qwen's corrected
fixpack verdict; MiniCPM +2.5 steps real against +0.6
sham). Onset is where they part: Qwen's onset pastes
moved their own margin separably, while MiniCPM's
onset margins barely differ between instances, so its
onset specificity test cannot be run to a verdict.
The naming transplant, the strongest effect in Qwen's
chapter (24 of 24 against 0 of 24 with the
both-matched control), reproduces in MiniCPM on the
clean half of its recipients (9 of 9 at the middle
entries against 0 of 12), with the other half already
naming the low tone beep-like before any paste.

**Mediation (cell 6).** Both: the corrected design's
verdict is the same. Restoring the recipient's own
post-sound rows at every layer from the paste up
undoes the transplant (Qwen at its four tested layers;
MiniCPM at every entry, deviations at about 5 percent
of the paste's move, with the restore-only validation
arm exact at zero). The effect is mediated by the
post-sound rows in both models.

**Probe cell (cell 7).** Both: the graded probe shows
the exact frequency traveling in the pasted state
while the margin only partially tracks within-band
differences (Qwen's lean moved about two thirds of its
gap; MiniCPM's moved 0.80 steps on the wide pairs).
MiniCPM adds one sharper edge: its probed frequency
still moves at paste entries 23 and 27 (0.87 octaves),
past the depth where its pitch answer stops responding
to pastes, so the carried-but-unused point is measured
directly in this chapter rather than inferred.

## What the comparison supports, and what it does not

Supported: the reporting-loss account of these models'
sound failures is not a one-model accident. Two
different open models, asked the same questions through
their own doors and opened with their own conventions,
show the same anatomy: the sound is in the state, the
ranking is in the margin, the transplant works inside
a window, the relay runs through the post-sound rows,
and the spoken answer follows the question's form.
The window closes at the same relative depth in both.

Not supported: any claim that the two models are the
same inside. Their margins differ in quality, their
onsets differ in strength, their default answers
differ by form, and MiniCPM's naming baseline for low
tones is split in a way Qwen's was not. The protocol
was held constant so the differences would be the
models' own, and they are.

## Records

MiniCPM reads: minicpm-behavioral-read.md,
minicpm-surgery-read.md, minicpm-proppatch-read.md,
minicpm-mediation-read.md, minicpm-probecell-read.md,
minicpm-tracing-read.md, and the codec read in
../codec-control/ (all in this workspace's
project-k-pilot tree). Run reports:
../run-reports/ (codec-minicpm, tracing-minicpm,
behavioral-minicpm, surgery-minicpm, proppatch-minicpm,
mediation-minicpm, probecell-minicpm). Builders and
notebooks: ../colab/. Journey: the Series Three journey
file's change log for Nov 7 to 10, 2026.
