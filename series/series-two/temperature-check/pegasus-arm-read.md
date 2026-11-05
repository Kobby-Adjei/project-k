# Pegasus temperature arm: the read (Nov 5, 2026)

Question: every Project K Pegasus answer on file was taken
at temperature 0. If the model is warmed up, does the sound
come back? This arm re-asked the claim-carrying cells at
temperature 0.7, 3 draws per cell, same files and prompts
byte-identical to the original runs. 30 calls, all HTTP 200.
Runner: run_tempcheck_pegasus.py on the adjeikwabena359
account (connector custom.twelvelabs-temp4, key
project-k-tempcheck-5). Raw answers:
tempcheck_pegasus_results.csv + raw-pegasus/.

## Score against the registered predictions

Swap files, plain sounds question (P3):
- silence: called silent 3 of 3. Control holds.
- speech: the sentence reported exactly 3 of 3, timed
  2.16 to 4.64 s, with "no other sounds" stated. Control
  holds.
- beeps: reported 0 of 3. All three draws say silent.
- water: reported 0 of 3. All three say silent.
- vacuum: reported 0 of 3. All three say silent.
- sizzle: reported 0 of 3. All three say silent.
- mixed: speech reported 3 of 3, bent the same way as at
  temperature 0 ("blue cattle" for "blue kettle"); the bed
  sound under it reported 0 of 3.

Real clips, repaired JSON prompt (P1), heard field:
- P02_07_5 (vacuum): "none" 3 of 3.
- P32_01_54 (jar under running water): "none" 3 of 3.
- own-01 (eggs): "none" 3 of 3.

Every registered prediction held. The failure condition
(any non-speech swap sound reported in at least 2 of 3
draws) did not come close: 0 of 3 in all four cells.

## What warming changed, and what it did not

At 0.7 the answers get wordier (full sentences instead of
one-word denials) and stay identical in content: speech is
reported and timed; everything that is not speech is
reported as silence or "none". Temperature moves the
wording. It does not move the hearing. The scoped claim
stands unchanged at temperature 0.7: through Twelve Labs
Analyze, what Pegasus 1.6 reports hearing is what a speech
transcript carries; non-speech sound does not reach its
answers.

## Scope of this read

One model (Pegasus 1.6 via Analyze), one warm setting
(0.7), 3 draws per cell, 10 cells. Temperature 1.0 was not
run on Pegasus; the Qwen 3B part (0.7 and 1.0) stays parked
by Kobby's Nov 5 ruling. No Series One text needs rescoping
to temperature 0 on the strength of this arm.
