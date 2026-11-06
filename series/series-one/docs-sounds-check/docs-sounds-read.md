# Docs sounds check: read (Nov 6, 2026)

## What ran
The three sounds the Twelve Labs models overview page names
in its Sounds row (chirping birds, applause, fireworks
popping or exploding), each a natural public-domain recording
(see provenance.md), cut to 10 s, loudness matched to about
RMS -20 dB, on the swap still picture and on a plain gray
screen. Pegasus 1.6, Analyze upload door, temperature 0,
33 calls, all HTTP 200. Answers read from the raw records
and the results CSV by Muse, not from the run log alone.

## Results
- Hearing question (P3), still picture, 3 repeats per sound:
  all 9 answers are the same sentence: "The audio track for
  this video clip is silent." Birds: 3 of 3. Applause: 3 of 3.
  Fireworks: 3 of 3. No sound was reported in any draw.
- Yes/no, sound present: 12 answers (each sound on still x2
  and on gray x2). All 12 are No. Birds No 4 of 4, applause
  No 4 of 4, fireworks No 4 of 4.
- Controls: the shared silent file asked about each sound,
  No 6 of 6 (correct). Wrong-sound lures on the still files,
  No 6 of 6 (correct).
- Total: 21 answers where a named non-speech sound was
  present and playing; the sound was reported in 0 of them.
  All 12 answers where the named sound was absent were
  correct No answers.

## Against the predictions
All three registered predictions held exactly: no P3 report
(9 of 9 silent), all matched yes/no answers No (12 of 12),
controls and lures clean (12 of 12).

## One detail worth keeping
Four of the applause and birds answers do not stop at "No."
They add: "The video consists entirely of a static gray
circle on a black background with no audio or visual cues
indicating any sound." The picture is described right. The
sound that is playing is still denied.

## What this does and does not show
It shows the reporting pattern from the earlier batteries
covers the docs page's own three example sounds, in these
recordings, through this door, at temperature 0. It does not
show what the model holds inside, and it is one recording per
sound. The applause recording's source tags mention talking
in the crowd (see provenance.md); no answer on the applause
cells mentioned voices, and the applause was denied like the
others, so the caveat does not change the read.

## Files
plan.md (predictions, written before the run), provenance.md,
measurements.json, build_docs_sounds.py, run_docs_sounds.py,
docs_sounds_results.csv, raw/ (33 per-call records).

## Independent replication (Nov 6, 2026)
Claude ran the same package in his own environment, on the
tests-only account key, before seeing this read. His 33
answers match ours cell for cell: the same silent sentence
on all 9 hearing answers, No on all 12 present-sound yes/no
answers, 6 of 6 muted controls and 6 of 6 lures correct,
and the long "static gray circle" sentence in the same
three applause-target cells, word for word. His note: the
answer tracks the question and the picture, not the sound;
the same long sentence comes back with the sound on and
with the silent file. His reply is saved verbatim at
claude-replication-2026-11-06.md.
