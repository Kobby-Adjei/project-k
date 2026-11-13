# Project K: Where did the sound go?

I study when multimodal AI systems fail, and I build the
evaluations that catch it. Project K is my failure
attribution work on audio-visual models: systems that are
given video with sound, and asked to report what happens.

The question behind every test here is simple. A model is
shown a clip with a clear sound in it. The sound does not
reach its answer. Where did it go?

## The tests

Every model in this project gets the same files and the
same questions. One thing changes at a time.

- The mute test: the same clip with its sound, then muted.
  If the answer does not change, the sound was never used.
- The swap: one still picture, many soundtracks. The answer
  has to follow the sound, not the picture.
- The bed: a spoken sentence laid over a second sound. Does
  the second sound bend the report, or vanish from it?
- The blackout: the sound stays, the picture goes black.
  What was the hearing, and what was the eyes helping?
- Yes or no, with a muted control: a forced choice, scored
  where the model exposes a score, and never trusted
  without its muted twin beside it.

## What has been found so far

Through the Twelve Labs Analyze routes, what Pegasus 1.6
reports hearing is what a speech transcript carries.
Non-speech sound, running water, a vacuum, cooking sounds,
a plain tone, does not reach its answers, under five
different ways of asking, while a spoken sentence in the
same clips comes back word for word. The full receipts are
in series/series-one.

Four open models ran the same batteries. Each lost sound in
a different place. MiniCPM-o 2.6 transcribes speech
perfectly and reports that it heard none. OLA-7B invents
sounds and scenes on pure silence, while the score under
its yes moves the right way. Qwen2.5-Omni-7B denies speech
as a sound and cannot name sounds its forced-choice answers
confirm. Qwen2.5-Omni-3B hears speech well until one still
picture is added, and then its read collapses. Three of the
four renamed a vacuum cleaner a door. The reads are in
series/series-two.

These are measurements, stated at the scope they were
measured at.

For one of the open models, the inside is now measured
too. Series Three opens Qwen2.5-Omni-3B and traces the
sound through its layers. The sound is present in the
model's inner states at every layer; what fails is the
reading of it. On synthetic tones, the spoken yes or no
is a fixed cut on an inner lean that ranks the truth
almost perfectly, and the cut is misplaced. That tones
result is scoped to synthetic tones under the tested
scoring; on real sounds the ranking tangles and the tone
cut does not transfer (measured in solution-tests/). Pasting one clip's inner states
into another clip's run moves the answer in a window of
layers, and in the deep end of that window the effect
reaches the answer only through the words that follow
the sound. Every claim is scoped to that one model. The
ledger, the protocol, and the frozen results are in
series/series-three.

## How this repo is organized

The method is permanent and lives in method/. Each round of
testing is a series, and each series gets its own folder
under series/, with its plan, its reads, and its result
files. New series slot in beside the old ones without
reorganizing anything. Attempts to fix what the series
found, each with its design and pass bar frozen before it
ran, live in solution-tests/. The notebooks that ran the
open models are in notebooks/. Data provenance is in
data/.

## What is not in this repo

No media files. The kitchen clips come from EPIC-KITCHENS
(CC BY-NC 4.0) and my own recordings are private; the file
recipes, measurements, and SHA-256 fingerprints are public
instead, so anyone can rebuild the exact files and check
them. No API keys; the runners read credentials from the
environment.

## Links

- Method and receipts for the Pegasus tests, quick version:
  https://gist.github.com/Kobby-Adjei/af49ec9ee67403e61971575e49cf123e
- The LinkedIn series "Where did the sound go?" is linked
  here as each post publishes.

## Status

Series One battery: complete for three open models; the
fourth (OLA-7B) is built and waiting to run. Series Two:
complete for all five models; the Pegasus text version of
the yes/no battery finished Nov 5, 2026 (60 of 60 answers),
and a temperature addendum (the same asks re-run at
temperature 0.7) sits beside it. On Nov 6, 2026 the three
sounds the Twelve Labs models page names as examples
(birds chirping, applause, fireworks) were run the same
way and denied the same way; an independent rerun matched
all 33 answers (series/series-one/docs-sounds-check).
Two follow-up check passes ran Nov 10, 2026: blurring the
picture in stages moved the timing answers and left the
naming flat, and a competing spoken sentence left the
timing and naming untouched while taking over the hearing
report completely, 30 of 30 (series/series-one/ross-checks).
An audio-only check of the two OpenAI audio models on
OpenRouter ran Nov 11, 2026; flat denial is not universal,
and the failure kind differs by model family
(series/series-two/openai-sounds-check). Series Three:
complete and frozen for Qwen2.5-Omni-3B; the MiniCPM-o 2.6
chapter ran under the frozen protocol Nov 7 to 10, 2026,
all seven cells, and sits beside it as an addendum
(series/series-three/minicpm-chapter). Solution tests:
four no-retraining fixes were tested in November 2026
under bars frozen before each run, and all four failed;
no fix is claimed (solution-tests/). Next new model in
the queue: Baichuan-Omni 1.5.

Kwabena Adu Adjei. Computer Science, Eastern Michigan
University. Thesis topic: Failure Attribution in
Audio-Visual Multimodal Models.
