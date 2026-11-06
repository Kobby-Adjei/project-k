# Docs sounds check: birds, applause, fireworks (Nov 6, 2026)

## Why this test exists
A friend of Kobby's pointed at the Twelve Labs models overview
page. Its table of what the models can identify has a Sounds row:
"Chirping (birds), applause, fireworks popping or exploding."
Those three exact sounds were never run through Pegasus 1.6.
The earlier batteries covered vacuum, running water, sizzling,
beeps, speech, and silence. This check runs the docs page's own
three examples, the same way the earlier sounds were run.

## Method (same shape as the swap and yes/no batteries)
- Three natural recordings, public sources, logged in
  provenance.md with author, license, and page link.
- Each sound cut to exactly 10.000 s, loudness matched to about
  RMS -20 dB, the same target the swap build used.
- Two picture versions per sound, built like the earlier ones:
  (a) the swap still picture (plain dark frame, light circle);
  (b) a plain gray screen (0x808080), the Series Two recipe.
- The muted control is the same shared silent file as before,
  swap_silence.mp4.
- Model pegasus1.6, POST /v1.3/analyze, temperature 0, through
  the temp4 connector (the adjeikwabena359 account, tests only).
- Questions:
  - Hearing form P3, word for word from run_swap.py:
    "What sounds can you hear in this clip? Describe only what
    is audible, without guessing from the pictures. If the
    audio is silent, say 'silent'."
  - Yes/no form, word for word from the Series Two battery:
    "Is there the sound of {target} in this clip? Answer yes
    or no." Targets: "birds chirping", "applause", "fireworks".

## Calls (33)
- P3 on the still version: 3 sounds x 3 repeats = 9
- Yes/no per sound: own target on still x2, own target on
  gray x2, own target on the muted control x2, one wrong
  target on still x2 (birds asked applause, applause asked
  fireworks, fireworks asked birds chirping) = 8 per sound,
  24 total.

## Predictions, written before any call (Nov 6, 2026)
1. P3: none of the three sounds is reported. All 9 answers say
   silent, none, or words to that effect.
2. Yes/no with the sound present (still and gray): all 12
   matched answers are No.
3. Muted controls: No 6 of 6 (correct). Wrong-target lures:
   No 6 of 6 (correct).
If a sound is reported, that is the result: the pattern from
the earlier batteries does not cover that sound, and the read
says so plainly.

## Scope
One model (Pegasus 1.6), one door (Analyze, upload), one
recording per sound, temperature 0. This tests reporting:
whether the answer names the sound. It does not test what the
model holds inside.

## Status
DONE Nov 6, 2026. 33 of 33 calls HTTP 200. All three
predictions held exactly. Read: docs-sounds-read.md.
Council replication asked Nov 6, 2026: package (clips,
scripts, plan, provenance; our results NOT included) on
Drive folder 16h-Lt-EgFLao1X9c_r5-qXMBSdwBBStT; ask posted
to #agent-handoff, ts 1793945409.873079. Claude runs it in
his own environment once Kobby pastes a fresh key for the
tests-only account into that thread himself (the stored
key's value is not readable by Muse by design).
Replication DONE Nov 6, 2026 (~2:00 AM EST): Claude's
independent run matched ours on all 33 answers, including
the long-form cells word for word. Reply saved at
claude-replication-2026-11-06.md. Claude's closing line:
the council key can be deleted now.
