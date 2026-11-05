# The temperature check (plan, Nov 5, 2026)

Kobby's question: every Project K run so far sat at
temperature 0 (Pegasus calls) or greedy (open models). What
if the sound reports live in the natural-temperature draw?
This check answers that on the protagonist model.

Lane: a Series Two addendum. It does not touch the locked
87-ask battery or its results; it re-asks a few of its cells
under sampling. Model: Qwen2.5-Omni-3B, fp16, transformers
4.52.4, free kadjei1 Colab, L4. Same files, same prompts,
byte-identical to the source runs. One question for every
draw inside a cell; only the sampling changes.

## Cells (8 asks)

Section A (Series Two 2x2, swap_speech.mp4, series2 runner
machinery):
- A1 sound alone: "Write down any words spoken in this
  clip." Greedy answer on file: the full sentence.
- A2 sound alone: "Is there a person speaking in this clip?
  Answer yes or no." Greedy: Yes.
- A3 sound + one still picture: transcription ask. Greedy:
  "742".
- A4 sound + still: presence ask. Greedy: No.

Section B (equal battery Part A, repaired JSON prompt,
equal runner machinery, video door):
- B1 jar, original: heard "water running from the faucet"
  (the catch; positive control).
- B2 vacuum, original: heard "a door is opened and closed".
- B3 jar, blackout: heard "no sound".
- B4 vacuum, blackout: heard "no sounds".

## Arms per ask (11 generations)
- Greedy x1 (in-session anchor; must reproduce the filed
  greedy answer, else the run stops and reports).
- Temperature 0.7 x5 draws.
- Temperature 1.0 x5 draws.
Sampling: do_sample on, temperature as stated, top_p 1.0,
top_k off. Seed per draw, logged in the row. Token caps
exactly as the source runners set them. Per-ask feed proof
with a hard stop, same as the source batteries.

## Registered predictions (written before the run)
- P1: A3 stays collapsed in most draws at both temperatures
  (full sentence in at most 1 of 5 at 0.7, 2 of 5 at 1.0);
  A4 stays No in most draws.
- P2: B3 and B4 keep denial as the common answer at both
  temperatures; the true sound is reported in at most 1 of
  5 draws per cell.
- P3: B2 names "vacuum" in at most 1 of 5 draws at either
  temperature; wrong names or denials dominate.
- P4: controls hold: A1/A2 and B1 report the sound in at
  least 4 of 5 draws at both temperatures.
If P1 or P2 fails clearly, the temperature objection stands
and the spoken-answer claims get rescoped to greedy in
every public text, starting with the posting map.

## Scoring
A draw "reports the sound" when: transcription returns the
full sentence or most of it; presence answers Yes; the JSON
heard field names the sound that is in the clip (close
synonyms count; judged in the read, raw answers saved so
the judgment is checkable). Report rate per cell per
temperature is the headline number.

## Part 1 status (Nov 5, 2026): PARKED by Kobby's ruling

The Qwen notebook was built and verified (12 cells, Drive id
1e2y5BhUdP3Grr1wcHH4uy_q4u5yVPIfN) and its run was started,
then stopped minutes in on Kobby's word: the temperature
goal is the Pegasus claims, and a Qwen result does not
answer for Pegasus. The notebook stays on Drive, unrun to
completion, as a parked extra. Part 2 below is the test.

## Part 2: the Pegasus arm (designed Nov 5; allowance probe
## answered HTTP 200 on Nov 5, 2026, so it runs)

PROBE (Nov 5, 2026, ~8:57 AM EST): one Analyze call on the
working key (the kobbytti account) returned HTTP 200 with a
full answer. BUT the arm's first call, minutes later,
returned HTTP 422 workload_allowance_exhausted ("overage is
disabled"), and the runner stopped by design. Reading of the
two facts together: the pool let one call through and then
shut; for a 30-call arm the allowance is effectively
exhausted. Kobby's conditional override ("if it's a new
account use my other emails") therefore TRIGGERS. The
console sign-in failure (kadjei1 password rejected) is
explained by the connector doc: the working key's account
is kobbytti@gmail.com, not the school one.

ARM (locked Nov 5, 2026): temperature 0.7, 3 draws per cell,
30 calls. Swap files under P3 (the plain sounds question,
byte-identical from run_swap.py): silence, speech, beeps,
water, vacuum, sizzle, mixed (21 calls). Real clips under
P1 (the repaired JSON prompt, imported from run_eggs.py so
it is exact): P02_07_5 (vacuum), P32_01_54 (jar, running
water), own-01 (eggs) (9 calls). The API takes no seed;
draws are repeats. Temperature-0 answers are already on
file from the original runs and are the comparison.

Registered predictions: speech swap reported in at least
2 of 3 draws (control); silence called silent in at least
2 of 3; beeps, water, vacuum, sizzle each reported in at
most 1 of 3; all three real clips keep denial or a miss as
the common answer (at least 2 of 3). If any non-speech swap
sound is reported in at least 2 of 3 draws, the temperature
objection has teeth and the Series One claims get rescoped
to temperature 0 in every public text.

Kobby's correction the same morning: the doubt bites hardest
on Pegasus, whose public claims rest on spoken answers only
(we never see its scores), and whose temperature dial was
ours: the pilot runner sets temperature 0 in the API call.
The arm: the Pilot Two denial cells (the real clips under
the repaired JSON prompt and P3, plus the swap files), asked
again at temperature 0.7, 3 draws per cell, report rate
scored the same way as Part 1. Prediction: the denial stays
the common answer; anything else rescopes the Pegasus
claims to temperature 0 in every public text.

(The earlier blocked note is superseded by the probe
result and the locked arm above.)

## Part 2 status (Nov 5, 2026): COMPLETE

The kobbytti pool let one probe call through, then shut
(422). On Kobby's explicit override of his Nov 2 no-new-
account rule, the arm ran on his pre-existing
adjeikwabena359@gmail.com Twelve Labs account (it already
existed; sign-up was refused "The user already exists";
Free plan, Workloads 0% used, allowance resets Nov 26,
2026). Serving key: project-k-tempcheck-5, stored in
connector custom.twelvelabs-temp4. Three earlier keys'
stored copies failed verification (copies taken through
the streamed browser window were mangled; the copy made
in Kobby's own browser worked first try).

Run: Nov 5, 2026, 30 of 30 calls HTTP 200, exit 0.
Verdict: every registered prediction held; no non-speech
sound was reported in any draw (0 of 3 in each of beeps,
water, vacuum, sizzle); all three real clips kept
heard "none" 3 of 3; controls held. The temperature
objection does not stand at 0.7; no Series One text needs
rescoping to temperature 0. Read:
pegasus-arm-read.md.

## Outputs
tempcheck_results.csv (one row per generation: cell, ask,
arm, draw, seed, sampling config, raw answer, feed proof),
run log, both saved local + Drive folder "temperature-check".
Read written from the CSV only.
