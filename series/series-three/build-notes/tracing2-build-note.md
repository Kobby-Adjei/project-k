# Tracing round 2: build note (Qwen 3B)

Built Nov 3, 2026. Builder: `colab/build_tracing2_qwen_notebook.py` (forked from
the round 1 builder). Notebook: `colab/project_k_qwen_tracing2_colab.ipynb`
(11 cells), uploaded to the Drive folder tracing-qwen3b (folder id
1n10NHxeMKyMmWKsbWvObe7IcdR88uJKR) as file id 1zeVC6BtnFKR5x0V4tp6IjphJQNj8kFT8,
73,663 bytes, size verified against the local file after upload. The notebook
was NOT run here; the run launches separately through the browser, on the free
kadjei1 lane, same as round 1.

## What round 2 changes (the round 1 read's fix list, followed exactly)

Round 1's readout was voided by its own rule: the shuffled-label control
sometimes scored perfectly, because 64 training clips against a 2,048-number
layer vector let the helpers memorize. Round 1's gate also found only 1 audio
token in a 287-position row. Three fixes, no others in spirit:

1. **Token check first, as its own early step with a hard stop.** For one tone
   clip and one speech clip (the codec masters), the runner prints the full
   histogram of token ids in the processed input row (every id with count > 0,
   token name and decoded form, special tokens flagged), the processor's audio
   feature lengths (input_features shape, feature_attention_mask true-frame
   counts), and every candidate audio id with its count in the row. The pooling
   id is derived from the histogram: the candidate with the highest count, the
   one at audio-frame scale, never the first candidate that merely occurs
   (round 1's mistake). Both clips must derive the same id and its count must
   be at least 10, or the step prints TOKEN CHECK FAILED and exits (code 11)
   before any extraction. One thinker forward on the tone clip then fixes the
   layer count and hidden size. The full transcript is saved either way as
   `tracing2_token_check.txt` in the tracing folder.
2. **Bigger battery, generated in-notebook, fixed seed 20261104.** Three
   classes, 360 clips each (1,080 total), 4.0 s at 16 kHz, raw WAV fed
   directly (no MP4, no codec anywhere, the codec control's raw arm form).
   Variation: tone bursts (0.5 s) with frequency jitter 950 to 1050 Hz, start
   0.5 to 3.0 s, level -24 to -14 dBFS; speech rendered from the probe-sentence
   master ("The blue kettle boils at seven forty two.", the same master file
   as round 1) with gain -6 to +3 dB, tape-style speed 0.92 to 1.08 (which
   shifts pitch with it), and start offsets up to 1.5 s; silence is no
   foreground at all. Every clip in every class sits on the same background
   scheme, digital silence with a one-in-six chance of a -75 to -65 dBFS
   noise floor, so the silence class is true digital silence plus near-silence
   noise-floor variants and the background cannot be a class cue. Every clip's
   per-second RMS and waveform SHA-256 are printed in the generation log (the
   feed proof), and the manifest stamps each clip's parameters, hash, and
   split. Split fixed by seed: 240 train / 120 held-out test per class, the
   same split for every layer and both tasks, re-derived in the probe step
   and asserted against the manifest stamps.
3. **Smaller, regularized readout.** Per layer, hidden states mean-pooled over
   the verified audio positions only. Two readouts, both reported in
   `tracing2_probe_results.csv`: (a) logistic regression, C=0.01, on
   standardized features; (b) the same probe after PCA to 64 components.
   Tasks unchanged: tone vs silence, speech vs silence. For every layer, task,
   and readout the CSV carries the real test accuracy and the shuffled-label
   control as the mean of 5 shuffles trained and tested identically (standard
   deviation beside it). The attention measure is kept: mean attention from
   the final position to the verified audio positions, now for all layers, on
   the round 1 subset shape (first 6 tone and first 6 silence clips), saved as
   `tracing2_attention.csv`. Features are saved as float16 in
   `tracing2_features.npz` with the file size printed (expected about 164 MB:
   37 hidden tensors x 1,080 clips x 2,048). Full candidate tracebacks go to
   `tracing2_load_log.txt`. The model still loads in eager attention, because
   the fast kernel returns no attention weights. Exit codes: 3 stack, 4 load,
   7 feed, 9 clip measurement, 10 extraction count, 11 token check.

## Verification results (local, before upload)

- Builder runs clean and writes the notebook (11 cells).
- Every code cell in the notebook parses; the embedded runner source parses.
- Battery counts are exactly 360/360/360 (one `range(360)` loop per class)
  with a 240/120 split: `order[:240]` train / `order[240:]` test per class,
  and a standalone simulation of the split function confirms 240 train / 120
  test, disjoint, per class.
- Prompts byte-identical to round 1, checked by evaluating both notebooks'
  PROMPTS dicts: tone "Is there the sound of a tone in this clip? Answer yes
  or no.", speech "Is there the sound of speech in this clip? Answer yes or
  no." (silence clips carry the tone prompt, as round 1's negatives did). The
  probe sentence appears verbatim in the notebook text.
- Grep of the built notebook confirms the token-check hard stop
  (TOKEN CHECK FAILED, exit 11, histogram built with Counter over the input
  row) and the 5-shuffle control (a `range(N_SHUFFLES)` loop with
  N_SHUFFLES = 5, `shuffled_accuracy_mean` in the results schema) are in the
  code, not just intended.
- No em dashes anywhere in the notebook (house rule).

## What to watch in the run

Check mode must show the derived audio id occurring at frame scale (dozens to
hundreds of positions) on both masters; anything near round 1's count of 1
means the check fails loudly, by design. Battery mode: 1,080 generated and
measured, 1,080 extracted, then 148 probe rows (37 tensors x 2 tasks x 2
readouts) with controls beside every score.
