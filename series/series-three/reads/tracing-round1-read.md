Series Three, test 2 (tracing), round 1. Series named Nov 3, 2026.

# Tracing round 1 (Qwen 3B): run complete, readout VOID by our own rule

Run Nov 3, 2026, free kadjei1 lane, L4, transformers 4.52.4. Check exit 0, battery
exit 0, 144/144 clips extracted, tracing_probe_results.csv 74 data rows (37 hidden
tensors x 2 tasks), tracing_attention.csv 48 rows. No errors anywhere. Local copies
of the CSVs and load log in this folder; features npz (43 MB) and clips on Drive in
tracing-qwen3b (folder id 1n10NHxeMKyMmWKsbWvObe7IcdR88uJKR).

## What the numbers say, and why we do not believe them yet

- Probe accuracy: 0.50 at layers 0 and 1, then 1.00 at every layer from 2 to 36,
  for BOTH tone and speech. The notebook's tail line reads "first layer below 0.70:
  never" for both tasks.
- The shuffled-label control, which the plan required to sit at chance, instead
  ranges 0.125 to 0.8125 (tone) and 0.0 to 1.0 (speech). A scrambled helper once
  scored a perfect 1.0. The pre-registered rule in tracing-plan.md: a control that
  beats chance voids the run until fixed. This readout is therefore VOID as
  evidence about layers.
- Why it happened (the design flaw, named): 64 training clips against a 2048-number
  vector per layer. A logistic helper with far more numbers than examples can
  memorize shapes instead of reading them, so both the real and the scrambled
  helpers can land perfect scores. The clean step at layer 2 is what that failure
  looks like, not a finding.
- Second suspect, unverified: the sanity gate's audio-token count was 1, in a
  sequence of 287 positions, for a 4-second clip. In this model family the audio
  normally occupies many positions. The gate picked the candidate id that decodes
  to <|audio_pad|> and occurs at least once; it did not check whether another
  candidate occurs far more often. If the pooled features came from a single
  marker position, the probes were reading the stream state at that slot, not the
  audio positions. This needs a printed input_ids histogram to settle.
- Attention, for what it is worth under the same cloud: mean attention from the
  final position to the audio position in layers 32 to 35 is 0.002 to 0.006,
  near zero, in line with the "barely looked at" prediction. Not usable until the
  position question is settled.

## Verdicts against the registered predictions

- Tone fades mid-to-late: NOT established (void readout).
- Speech readable at every layer: NOT established (same void).
- Controls at chance: FAILED, which is what voids the round.
- The machinery itself (notebook, gates, extraction, file discipline) worked end
  to end on the first run. What failed is the probe design's sample size and,
  possibly, the token choice.

## Round 2 fix list (for Kobby's go)

1. Settle the token question first: one short check run printing the histogram of
   token ids in a real input row, naming the audio positions and their count.
2. Many more probe clips (hundreds per class; generation is synthetic and cheap),
   a fixed large held-out set, and a smaller or regularized readout, so a helper
   cannot memorize. Report the shuffled control per layer beside every accuracy.
3. Re-extract only if the token answer changes the positions; otherwise the saved
   features stand and only the probe step reruns.
