# Solution tests

The series in this repo measure where the sound goes. This
folder asks the next question: can anything make the model
get the sound right, without retraining it?

My rule for these tests is the same as for the series. The
design and the pass bar are written down and frozen before
anything runs. A fix counts only if it clears the frozen
bar on sounds it was not built on. A test that fails is
kept here as a failure, with its files, and is never
described as a fix.

Four tests ran in November 2026. All four failed their
frozen bars. No fix is claimed anywhere in this folder.

## The tests

1. **wrapper-pegasus/** Ask about the sound alone first,
   then carry that answer into the final answer (Pegasus
   1.6). Failed: the sound-only step answered "The audio is
   silent." on running water, on a vacuum, and on applause,
   the same words it used for real silence. There was no
   correct first step to carry forward, and no disagreement
   to flag. 28 calls.

2. **wrapper-minicpm/** The same wrapper on MiniCPM-o 2.6,
   an open model. Failed for the opposite reason: the
   sound-only step did carry signal (2 of 3), but asking
   again in ordinary words matched it (2 of 3). The wrapper
   added nothing over re-asking. 28 asks.

3. **inside-cut-minicpm/** Series Three found that on
   synthetic tones, the model's inside scores rank high
   tones above low ones almost perfectly, and only the
   yes/no cut sits in the wrong place. This test carried
   the tone cuts, unchanged, to real sounds. Failed: the
   ranking itself tangles on real sounds. Sounds that are
   present (a vacuum, speech) scored below silence. Yes
   share AUC 0.7222 against a frozen kill line of 0.80,
   and no cut placement on this set scores above 13 of 18.
   Scope note: the Series Three tones claim holds for
   synthetic tones under the tested scoring. Tone AUC shows
   recoverable low-level acoustic features, not semantic
   identification of real sounds, and this transfer failure
   belongs beside the tones claim wherever it is stated.

4. **tagger-pegasus/** A pretrained audio tagger (CLAP,
   zero-shot) listens first and writes its label as text;
   the text goes into the video model's prompt, because
   text is what reaches its answers. The tagger alone
   scored 4 of 4. With true labels, Pegasus named 3 of 3.
   With deliberately wrong labels, Pegasus copied all 4,
   including reporting running water on a silent clip.
   Failed under the frozen bar (blind copies 4 of 4 against
   a maximum of 1). The hearing in that pipeline is the
   tagger's; the video model verifies nothing. See
   tagger-pegasus/FINDING-blind-copy.md.

## What each folder holds

The frozen design, the runner or notebook, the result
file, and the report, so every number above can be
recomputed from the files in this repo.
