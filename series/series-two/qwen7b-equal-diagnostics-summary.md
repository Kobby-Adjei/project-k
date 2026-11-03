# Qwen2.5-Omni-7B, equal battery read (Nov 2, 2026)

Source: equal_qwen7b_results.csv, 103,873 bytes, 41 data
rows, downloaded from the kobbytti clips folder by Drive
id 1VDtyUg4pUm91cmy8IkzjlBbD5AkOWghg and read here. Run:
kobbytti Colab, L4 High-RAM, fp16, transformers 4.52.4,
Nov 2, 2026, 11:36 AM to 11:50 AM EST. Sanity answer
"One", check passed. Repeats are identical in nearly
every cell (greedy decode). Feed proof is in the CSV:
every row carries its measured track; the matched speech
track is the same track as the swap speech file (RMS
-20.0 dB, nonzero share 0.2778, 10.005 s).

## Part A, the repaired question, real clips and swaps

- Vacuum clip: heard "sizzling food, clinking utensils,
  pots and pans" in both repeats, start 0. No vacuum
  named. The sound it reports matches the kitchen it
  sees, not the vacuum the clip is built around.
- Jar clip: heard water running, sponge scrubbing,
  dishes clinking, start 0. The water is really there;
  a clean catch.
- Eggs (own recording): heard "sizzling and stirring",
  start 2. A clean catch.
- Swap silence: "None", clean.
- Swap speech: "None" in both repeats. The sentence is
  denied as a sound, the same shape as its Series Two
  result.
- Swap beeps: "beeping", start 4.0. Catch.
- Swap water: "water trickling". Catch.
- Swap vacuum: "none". Miss.
- Swap sizzle: "none". Miss.
- Swap mixed (speech over sizzle): "none". Miss.

## Part A2, muted and blackout pairs

- Vacuum muted: "None" both repeats. Clean.
- Vacuum blackout (real vacuum sound, black picture):
  "door opening and closing", start 0.00, both repeats.
  It reports a sound, but the name is invented. The
  same invented label the 3B gave this clip in its
  Series One probe; Kobby's ear verdict on this clip is
  still owed.
- Jar muted: "None". Clean. Jar blackout: "none" —
  with the picture gone it loses the water it caught
  with the picture on.
- Eggs muted: "none". Clean. Eggs blackout: "none" —
  same pattern; the picture was carrying the sizzle
  catch.

## Part B, the matched bed, two doors

- Plain question (P3) on the clean speech file:
  "silent." in both repeats. The baseline FAILS: the
  track is proven speech at -20 dB and the model calls
  it silent. Under the battery's own rule, this model's
  bed rows through this question are VOID: beeps bed
  "A beep is heard." (the beep caught, the speech still
  missed), vacuum bed "Silent."
- Transcription door ("Write down any words spoken...
  say 'no words'"): "no words" on all three files,
  including the clean speech baseline. Through this
  door, on these matched files, the model finds no
  words at all. Scope note: in Series Two, a different
  wording of the transcription ask on the swap speech
  file returned the sentence word perfect; the loss
  here is tied to this door and these files, not a
  blanket "cannot transcribe".

## The shape

Same model, sharper picture: on real clips it catches
water and sizzle and misses the vacuum, replacing it
with sounds from what it sees. Take the picture away
and two of its three real-clip catches vanish, while
the vacuum turns into an invented door. Speech is
denied in every form this battery tried: open report,
plain sounds question, and the transcription door.
Its Series Two label holds and widens: the loss sits
in the reporting, and the eyes steer the report.
