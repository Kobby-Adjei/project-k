# Graded probes: build note (Nov 4, 2026)

Council order item 3 (Claude's Nov 4 audit): step 5's binary contrasts
(500 vs 1500 Hz, 0.5 vs 2.5 s) sat at the ceiling, so a perfect score
could not show decay. This battery regresses the actual values.
Kobby's sequencing (Nov 4): graded probes next; the council gets
everything (index check + behavioral + graded) in one post after.

Notebook: colab/project_k_qwen_graded_colab.ipynb; builder:
colab/build_graded_qwen_notebook.py. Runner (46,307 chars, compiles;
the notebook's embedded copy was extracted and compile-checked) is the
property battery's machinery whole (proof, load, feed, token check,
generation discipline incl. write_clip always-overwrite + read-back,
extraction with resume), with generate_clips and the probe stage
replaced and output names switched property_ -> graded_.

Stimuli (generated in-notebook, seed as the property battery): 360
tones, 30 pitches log-spaced 300-2400 Hz x 12 onsets 0.25-3.25 s,
0.5 s bursts at -20 +-1 dBFS on the shared background scheme, pitch
jitter x0.99-1.01, onset jitter +-0.05 s, random phase. Exact jittered
values are the targets (manifest freq_hz, start_s). All 360 waveforms
asserted unique by hash. Splits: by clip 240/120 (seeded permutation,
stamped); held-out pitch computed in the probe stage (train on even
pitch grid indices, test on odd).

Readouts: per state 0-36, positions audio-mean and final; ridge
regression for log2(freq) (MAE in octaves) and onset seconds (MAE in
s); readouts pca64 (PCA-64 + Ridge alpha 1.0) and plain (standardized
+ Ridge alpha 100). Nulls: 200 target permutations, pca64, by-clip
split, focus states 16-25, reported as null R2 mean/p95 on the
matching rows. Output: graded_probe_results.csv (37 x 2 x 2 x 2 x 2
= 592 rows), graded_manifest.csv, graded_features.npz.

Predictions registered before the run: pitch MAE under ~0.06 octaves
and onset MAE under ~0.2 s at every state if graded fidelity matches
the binary result; final position at state 0 at chance (built-in
sanity); decay, if any, appears as MAE rising with state, final
position first.
