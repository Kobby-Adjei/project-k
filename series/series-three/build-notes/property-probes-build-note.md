# Handover Correction step 5: build note (Nov 3, 2026)

Property probes, per the council's Q3/Q4 designs and Kobby's order (before
any MiniCPM run). Builder: colab/build_property_qwen_notebook.py, derived
from the tracing round 2 builder. Notebook:
project_k_qwen_property_colab.ipynb, Drive tracing-qwen3b folder.

Stimuli (generated in-notebook, seed 20261105): factorial tones, pitch
500/1500 Hz x onset 0.5/2.5 s, 90 instances per cell (360 tones), burst
0.5 s at -20 +-1 dBFS, per-instance pitch jitter x0.98-1.02, onset jitter
+-0.1 s, random phase, on round 2's shared background scheme; 60 silence
clips. Every waveform hash asserted unique in the runner. Splits by
instance within each cell (60 train / 30 test per cell; silence 40/20),
stamped in the manifest (property_manifest.csv carries pitch_band and
onset_band).

Readouts: per state (37 states; state k = output of decoder layer k-1),
the audio positions mean-pooled and the final row position, each with the
plain and PCA-64 logistic readouts (C=0.01, as round 2). Tasks: pitch
(high vs low), onset (late vs early), presence (tone vs silence, audio
readouts only, 5-shuffle controls as round 2). Nulls: 200 permutations
for the two property tasks at states 16 to 25 (decoder layers 15 to 24),
PCA-64 readouts at both positions, reported as mean/std/p95 in
property_probe_results.csv. Features (audio + final) saved as
property_features.npz. Feed: round 2's picture + audio feed, so the
states are comparable to round 2's presence curves; the token check and
hard-stop discipline are round 2's unchanged.

## Run history and the stale-file bug (Nov 3, 2026, evening)

- Run 1 (original builder): stopped at the end of generation on the all-hashes-unique assertion. Most silence clips are identical digital silence by design, so the assertion itself was wrong. Fix: uniqueness required only for tones; silence grouped by waveform hash, each group entirely on one side of the split.
- Run 2 (fixed builder): extraction hard stop, exit 7; the failing clip could not be identified because Colab's output area would not scroll. Fix: the runner now tees all output to property_run_log.txt in the Drive folder, and the feed hard-stop banner prints an on-disk check of the failing file.
- Run 3 (diagnostic): the log named it. Extraction stopped at clip 413 of 420, prop_silence_53: the manifest row (from the in-memory wave) says a -67.54 dB noise floor, expect_silence False; the file on disk was all zeros. Root cause: write_clip copied an existing Drive file into place instead of writing the wave in hand. /content is fresh each session but the Drive property_clips folder persists across runs, and the draw sequence had moved between builder versions, so the same filename held a different wave. The 412 clips extracted before the stop are void (files may not match the manifest); no features file was saved, so nothing downstream is contaminated.
- Fix (run 4): write_clip always writes the wave in hand to both homes, overwriting; finish_row reads the written file back and stops generation if its RMS disagrees with the generated wave by more than 1 dB. Runner recompiled and verified; Drive notebook replaced in place.
- Note: the same copy-if-exists write_clip pattern exists in the tracing2 builder. Tracing round 2's outputs were verified against its manifest at the time (features reuse checks passed); leave it, but do not reuse that pattern in new builders.
- Run 4 (overwrite + read-back fix): COMPLETED clean Nov 3, 2026, 9:31-11:22 PM EST (browser task, L4, exit 0). Token check passed; 420 clips generated (360 unique tone waveforms; silence in 9 unique waveforms, each on one side of the split; tone 240 train / 120 test; silence 52 train / 8 test); 420 of 420 extracted; headline CSV saved before the nulls, as designed. Read: property-probes-read.md in this folder. Headline: pitch and onset both linearly readable at 1.000 at every state from 1 to 36 at the audio positions and at the final answer position (chance at state 0, as expected); 200-permutation nulls at the focus band average ~0.50 with p95 at most 0.718. The presence sanity line (flat 0.9375) is uninformative as run (only 8 silence clips in test; always-tone scores the same); the read says so. Step 5 is DONE; The Handover Correction is closed.
