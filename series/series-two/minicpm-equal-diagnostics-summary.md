# MiniCPM-o 2.6, equal battery read (Nov 2, 2026)

Source: equal_minicpm_results.csv, 65,086 bytes, 41
data rows, downloaded from the kobbytti clips folder
and read here. Run: kobbytti Colab, L4 High-RAM, bf16,
transformers 4.44.2 (its own pin), Nov 2, 2026, 12:09 PM
to 12:32 PM EST. Sanity answer "A cat sits on a mat."
exact. Repeats identical in every cell. Note: MiniCPM
wraps its JSON in a prose lead-in ("Based on the audio
you provided..."); the heard values below are read from
the JSON body of each answer.

## Part A, the repaired question, real clips and swaps

- Vacuum clip: heard "none". Miss.
- Jar clip: "Running water". Catch.
- Eggs: "none". Miss (quietest track caveat stands).
- Swap silence: "none". Clean.
- Swap speech: "none". The sentence denied as a sound,
  its standing shape.
- Swap beeps: "beeping sound". Catch.
- Swap water: "water dripping". Catch.
- Swap vacuum: "none". Miss.
- Swap sizzle: "none". Miss.
- Swap mixed: "none". Miss.

## Part A2, muted and blackout pairs

- Muted: "none" on all three. Clean; nothing invented
  on any muted clip.
- Blackout: "none" on all three. With the picture gone
  it reports no sound on any real clip, the jar's water
  included.

## Part B, the matched bed, two doors

- Plain question (P3) baseline, clean speech: "The
  audio is silent." Baseline FAILS, so its bed rows
  through this question are void. For the record:
  beeps bed "There is a beeping sound in this clip."
  (the beep caught, the speech missed); vacuum bed
  "a door opening and closing, followed by a sound of
  a machine or appliance running" (the door label
  again, on a vacuum-family sound; the speech missed).
- Transcription door baseline: "The blue kettle boils
  at 742." The true sentence is "The blue kettle boils
  at seven forty two", so this is the whole sentence
  returned, with the number written as digits. The
  baseline PASSES through this door. Beeps bed: "The
  blue kettle boils at 742." Vacuum bed: "The blue
  kettle boils at 742." Identical on all three files:
  neither bed bends this model's transcription at all.
  (Contrast Pegasus, where the vacuum bed bent its
  transcript while the answer denied other sounds.)

## The shape

Its Series Two label holds exactly: the reporting is
where sound dies. Detection shows through the direct
forms (beeps named in both batteries, water named),
the open report drops vacuum, sizzle, mixed, and all
speech, and the blackout takes away even the water.
One gain over Series One: its bed finally has a
readable door, and through that door the beds are
innocent.
