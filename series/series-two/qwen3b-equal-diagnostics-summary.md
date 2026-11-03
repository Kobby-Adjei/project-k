# Qwen2.5-Omni-3B, equal battery read (Nov 2, 2026)

Source: equal_qwen_results.csv, 102,233 bytes, 41 data
rows, downloaded from the kobbytti clips folder and read
here. Run: kobbytti Colab, L4 High-RAM, fp16,
transformers 4.52.4, Nov 2, 2026, 11:50 AM to 12:09 PM
EST. Repeats identical in every cell. Feed proof in the
CSV: the matched speech track is the same proven track
as the other models' (RMS -20.0 dB, nonzero 0.2778).

## Part A, the repaired question, real clips and swaps

- Vacuum clip: heard "a door is opened and closed",
  start 0.0. The same invented label as its Series One
  probe of this clip. No vacuum named.
- Jar clip: "water running from the faucet". Catch.
- Eggs: "none". Miss (the eggs track is the battery's
  quietest; the level caveat stands).
- Swap silence: "none". Clean.
- Swap speech: the answer is an empty shell, "**Answer:
  **" twice and nothing else. No words, no "none", just
  the empty form.
- Swap beeps: "a beep", start 2.41. Catch.
- Swap water: "a sound of a water tap running". Catch.
- Swap vacuum: "a sewing machine is running", repeated
  three times in one answer. Invented.
- Swap sizzle: "a sound of a machine working", start
  10.0. A machine, not a sizzle.
- Swap mixed: the answer collapses into number garbage
  (": 1940, 194, 194, ..."). A broken answer, saved as
  it came.

## Part A2, muted and blackout pairs

- Muted: "none" on all three. Clean; nothing invented.
- Blackout: "no sounds" or "no sound" on all three,
  including the jar, whose water it caught with the
  picture on. With the picture gone, this model hears
  nothing at all on the real clips. Its real-clip
  catches were carried by the picture.

## Part B, the matched bed, two doors

- Plain question (P3) baseline, clean speech: "silent".
  Baseline FAILS, so its bed rows through this question
  are void. (Beeps bed also "silent"; vacuum bed invents
  "a sewing machine, a door opening and closing, and
  music".)
- Transcription door baseline: "no words". This door's
  baseline fails too, so the bed is void through both
  doors for this model. Scope note: in Series Two, a
  different transcription wording on the swap speech
  file returned the sentence nearly whole with sound
  alone; the failure here is tied to this door and
  these files.

## The shape

The mirror of its Series Two self. There, its yes/no
answers were the cleanest of the four. Here, in open
report form, it invents (door, sewing machine), breaks
(number garbage, empty shell), and goes deaf the moment
the picture is removed. The form of the question
decides what this model can do.
