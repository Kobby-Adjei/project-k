# Feed audit: did sound reach MiniCPM-o? (Series Two, test 1)

Nov 2, 2026. Local audit, no GPU. This is the first Series Two
test, run at Kobby's direction ("audit our feed first, then OLA")
after the council read the four-model results.

## Why this audit exists

Two charges from the council replies of Nov 2, 2026
(council/council-replies-2026-11-02-four-model.md):

- ChatGPT's charge: the MiniCPM run never proved that sound
  reached the model on any given row. The sanity check only
  proves the model can write text. The notebook's audio
  extraction is not enforced as a hard per-row success condition,
  and a missing track can be silently filled with zeros. Until
  the pathway is verified, no answer of "none" can be blamed on
  the model with confidence.
- Claude's checks 4 and 5: no open model has ever been run on
  the swapped vacuum soundtrack or on true silence, so the
  false-alarm rate on silence is unknown; and MiniCPM was fed
  at most the first 15 seconds of every clip, so every target
  sound's window must be confirmed inside that span.

This audit does three things: reads the runner's audio path
line by line, measures the 12 files the run carried as decoded
sound, and gives a verdict per cell group.

## Part 1: the code path

The runner lives inside colab/build_minicpm_notebook.py as
RUNNER_SRC. The function that feeds every ask is build_units
(builder lines 778-812). Traced behavior:

1. Frames: ffmpeg extracts one frame per second
   (subprocess.run, lines 787-789) with check=False. The return
   code is never read. If frame extraction fails outright, the
   frame list is empty and every unit silently gets a blank
   black 448x448 picture instead (lines 803-807).
2. Sound: ffmpeg extracts the track to audio.wav, mono, 16 kHz
   (lines 790-792), again with check=False. The return code is
   never read, and nothing checks that ffmpeg said anything at
   all.
3. If audio.wav does not exist afterwards, the code substitutes
   an empty track with no warning printed anywhere (lines
   793-796: the else branch sets audio_np to np.zeros(0)).
4. If the track exists but runs shorter than the picture, the
   unit count follows the longer of the two (lines 798-800),
   and every missing second is padded with digital zeros to a
   full second (lines 808-811). A clip whose audio extraction
   failed completely would therefore be fed as a full chain of
   framed units carrying pure silence, indistinguishable in the
   output from a model that heard nothing.
5. Nothing about the sound is logged per ask: no extracted
   duration, no level, no nonzero count, no unit count. The
   results CSV has no audio column. The only per-ask prints are
   the parsed heard and seen fields.
6. What does fail loudly: a hard exception inside an ask (for
   example librosa refusing a corrupt wav) is caught by the
   battery loop and written into the CSV as an ERROR row with a
   RUN PROBLEM banner (builder lines 865-893). So a corrupt
   track would have shouted. A missing track whispers: it takes
   path 3 or 4 above and says nothing.
7. The sanity check is a text-only ask ("Reply with the word
   ready.", no picture, no sound; builder lines 697-730).
   ChatGPT's reading is exactly right: it proves the model can
   write, and nothing about sound reception.
8. One step earlier, the notebook's clips step (Step 2) checks
   only that each file exists and prints its size in MB. It
   never checks for an audio stream. Its one edit, cutting any
   file longer than 20 seconds down to its first 15, also runs
   ffmpeg with check=False (builder lines 477-492).

Verdict on the code: ChatGPT's charge stands in full. The path
has a real silent-failure mode, it was live for every ask of
the Nov 2 run, and the run left no per-row evidence that could
clear it after the fact.

## Part 2: the files, measured

Method: ffprobe for stream layout; each file decoded with
ffmpeg to mono 16 kHz float, the same extraction shape the
runner uses, and measured with numpy. RMS is in dB relative to
full scale. Per-second traces list each second's RMS; a reading
of about -120 dB means digital silence in that second. These
are the local source files the Drive clips folder was built
from; byte identity with the Drive copies is not claimed (see
the jar note under Part 3).

| File | Audio stream | Length | Overall RMS | Nonzero samples | Peak |
|---|---|---|---|---|---|
| P02_07_5.mp4 (vacuum, real) | AAC 48 kHz | 14.7 s | -25.0 dB | 100% | 1.25 |
| P32_01_54.mp4 (jar, real) | AAC 48 kHz | 15.0 s | -36.7 dB | 100% | 0.10 |
| own-01.mp4 (eggs, fed 15 s copy) | AAC 44.1 kHz | 15.0 s | -43.8 dB | 98.8% | 0.14 |
| probe_speech.mp4 | AAC 48 kHz | 8.0 s | -14.8 dB | 35.2% | 1.25 |
| probe_tone.mp4 | AAC 48 kHz | 8.0 s | -7.7 dB | 19.1% | 1.41 |
| swap_speech.mp4 | AAC 48 kHz | 10.0 s | -17.0 dB | 28.0% | 1.09 |
| swap_beeps.mp4 | AAC 48 kHz | 10.0 s | -17.0 dB | 15.8% | 0.54 |
| swap_water.mp4 | AAC 48 kHz | 10.0 s | -17.0 dB | 100% | 0.91 |
| swap_sizzle.mp4 | AAC 48 kHz | 10.0 s | -20.2 dB | 97.6% | 0.88 |
| matched_speech.mp4 (baseline) | AAC 48 kHz | 10.0 s | -17.0 dB | 28.0% | 1.09 |
| matched_speech_beeps.mp4 | AAC 48 kHz | 10.0 s | -14.0 dB | 35.8% | 1.34 |
| matched_speech_vacuum.mp4 | AAC 48 kHz | 10.0 s | -14.9 dB | 100% | 1.19 |

The raw measurement record is saved beside this audit as
feed-audit-measurements.json.

Two more files exist in the swap set and were measured because
Claude's check 4 turns on them: swap_silence.mp4 is true digital
silence (0.0% nonzero samples, peak 0.0, no signal at any
second), and swap_vacuum.mp4 carries a continuous vacuum track
(-19.0 dB overall, sound in every second). Neither file appears
in the MiniCPM job list, nor in the Qwen 7B battery. Only
Pegasus ever sat them, in the Nov 1 swap matrix.

Per-second RMS traces (dB, seconds 0, 1, 2, ...):

- Vacuum (real): -38.0, -31.0, -23.8, -35.8, -36.6, -33.3,
  -33.0, -31.2, -20.2, -16.6, -22.2, -31.5, -31.0, -37.9, -37.5
- Jar (real): -39.2, -38.7, -39.2, -39.6, -40.2, -38.1, -39.9,
  -38.6, -32.8, -30.0, -37.8, -39.2, -38.8, -39.1, -37.8
- Eggs (fed copy): -51.6, -45.1, -41.2, -40.0, -40.4, -40.6,
  -41.3, -41.0, -51.0, -48.6, -49.7, -49.8, -49.9, -50.3, -49.9
- Probe speech: silence, silence, -9.8, -9.1, -14.0, silence,
  silence, silence
- Probe tone: silence, silence, silence, -3.4, -3.5, -23.5,
  -3.4, silence
- Swap speech: silence, silence, -11.1, -10.4, -15.2, then
  silence to the end
- Swap beeps: silence x3, -11.8, -11.9, -28.8, -11.8, then
  silence
- Swap water: -20.3, -19.7, -20.2, -20.6, -21.2, -19.0, -21.0,
  -19.7, -13.8, -11.0
- Swap sizzle: -25.1, -22.3, -19.2, -18.2, -19.1, -18.5, -19.4,
  -19.4, -24.4, -22.1
- Matched baseline: identical trace to swap speech (same bytes
  in the track: sentence at seconds 2-4)
- Matched beeps bed: sentence at seconds 2-4 with the middle
  second lifted to -8.0 by the beeps, beeps again at second 6
  (-11.8), second 5 dipped (-28.8)
- Matched vacuum bed: -28.6, -21.7, -10.5, -10.4, -14.8, -23.7,
  -23.4, -21.7, -14.9, -13.0 (bed in every second, sentence at
  seconds 2-4)

The eggs provenance: the original own-01.mp4 is 58.8 s. Its
first 15 seconds match the fed 15 s copy to within 0.2 dB at
every second, so the fed copy is a straight cut of the
original's opening, and the trace above describes what was fed.

## Part 3: cell by cell, the window check (Claude's check 5)

The fed span is the first 15 seconds: the runner caps every
ask at 15 one-second units, and Step 2 pre-cuts anything over
20 seconds to 15. Every battery file's audio is 15 seconds or
shorter, and every target sound sits inside the span:

- Vacuum (real): the whole file is the span. Sound in every
  second, loudest at seconds 8-10.
- Jar (real): sound in every second of the local 15 s copy,
  loudest at seconds 8-10, and the water passage the Nov 1
  listening record places from about 5 s sits inside either
  way. One footnote: a Step 2 listing relayed during the runs
  put the Drive copy at 20.0 s, where the local source measures
  14.97 s. The discrepancy is unresolved (the relayed listings
  of that night bent before), and it does not move the verdict:
  under the 20 s reading the runner's cap still covers seconds
  0-15, which contain the passage.
- Eggs: the ignition and sizzle passage occupies seconds 2-7
  of the fed copy (about -40 dB per second), fully inside.
- Swap cells: all four tracks are 10 s files with their sounds
  at seconds 0-9 (water, sizzle) or 2-6 (speech, beeps).
- Matched cells: the sentence sits at seconds 2-4 in all three
  files, the beeps at seconds 3-6, the vacuum bed in every
  second.

Check 5 is settled: no target sound fell outside the window
MiniCPM was fed. The 15-unit cap cost nothing in this battery.

Check 4 is a proven gap, not a suspicion: swap_vacuum.mp4 and
swap_silence.mp4 exist, are measured above, and appear in no
open-model job list. For the open models, the false-alarm rate
on true silence is unknown, and "the vacuum was never named"
rests on the single real clip (and, for MiniCPM, on the vacuum
bed in Part B, also unnamed). Series Two test 3 closes this by
running the full swap set.

## Verdicts by cell group

**Real clips.**
- Vacuum, answered none: the file side is clean. A loud,
  continuous track fills the whole fed window. The code side
  keeps a narrow doubt alive: a one-file extraction failure in
  that session would have left no trace. Against it, the same
  code path in the same session delivered the jar's water and
  the swap water, both reported. A feed artifact is very
  unlikely, and cannot be reduced to zero after the fact.
- Jar, answered "Running water": a hit, and the audit's anchor.
  It proves the full path, file to ffmpeg to units to model,
  delivered a non-speech sound at least once in this battery.
- Eggs, answered none: not a missing-track artifact. The sound
  is in the file and in the window. But a level caveat stands
  and must ride beside this result from now on: the eggs track
  is the quietest material in the battery (-43.8 dB overall,
  sizzle passage about -40 dB), roughly 20 dB under the
  synthetic tracks' matched level. Whether the model's front
  end treats a -40 dB passage the way it treats a -20 dB one is
  a property of the model, not provable from the file. The
  honest sentence is: MiniCPM was fed a real but quiet sizzle,
  and reported none.

**Swap cells.**
- Speech, answered none: the file side is disproven as a
  cause. These are the same bytes Pegasus transcribed
  word-perfect through the API, with the sentence loud at
  seconds 2-4. Session-side doubt is the same narrow one as
  the vacuum, and Part B weakens it further (below).
- Beeps, answered none: loud, peaky beeps at seconds 3-6,
  inside the window. Same verdict shape as speech.
- Sizzle, answered none: a loud broadband track fills seconds
  0-9 at the matched level. Sound was there to be fed. Claude's
  caveat stands as a question about the sound's character, not
  its presence: the track was boosted by about 26.6 dB at build
  time, and a boosted sizzle may read as hiss. That is a
  stimulus question for a later test, not a feed gap.
- Water, answered "water dripping": the second delivery proof.

**Matched cells (Part B).**
- Baseline, answered "The audio is silent.": the strongest
  cell in the audit. The track is byte-identical to the swap
  speech track, the sentence sits loud at seconds 2-4, and the
  same bytes were transcribed perfectly through another door.
  "Silent" is false about the file. And the delivery doubt is
  at its weakest here: two asks later in the same part, through
  the identical path, the model named the beep on the beeps-bed
  file. Sound demonstrably reached this model in Part B.
- Beeps bed, beep named and speech not reported: both sounds
  are measured present in the file. The unreported speech is a
  fact about the answer, not about the feed.
- Vacuum bed, answered silent: the file carries the sentence
  at seconds 2-4 and a vacuum bed in every second. "Silent" is
  false about the file on both counts.

**Summary of the verdict.** No "none" or "silent" answer in
this battery can be blamed on a missing or empty file, an
absent audio stream, or a sound outside the fed window; the
measurements close all three. What remains open is only the
session itself: the runner's silent paths mean a per-file
extraction failure inside that Colab session would have left
no evidence, and none can be manufactured now. The two water
hits and the Part B beep make a wholesale feed failure
impossible and a selective one unlikely, but unlikely is the
strongest word the record supports. That is precisely the hole
Series Two test 2 is built to close, by printing the feed
proof from inside the session.

## What this audit cannot settle

- What actually entered model.chat on each ask in that
  session. The session is gone, its working files with it, and
  the runner logged nothing about the sound.
- Whether Colab's ffmpeg behaved on these files exactly as
  local ffmpeg did. Same tool, same flags, but the run never
  read a single return code, so even a total failure would
  have left the record exactly as clean as success did.
- Whether the Drive folder's copies are byte-identical to the
  local sources measured here. The jar duration discrepancy
  above is the visible tip of that gap.
- Anything inside the model's own preprocessing once the
  chunks were handed over. That is the model's side of the
  door. Series Two reaches it by intervention (sound alone,
  yes/no scoring), not by auditing our side harder.

## What the Series Two diagnostic run must print

From inside the session, per ask, from the model's own feed
path, saved beside the results:

1. The ffmpeg return code for both extractions (frames and
   track), per clip.
2. The extracted track's duration, sample count, overall RMS,
   and nonzero share, computed on the exact array the units
   are built from.
3. Per-unit chunk levels: the RMS of every one-second chunk
   actually handed to the model, or at minimum the count of
   all-zero chunks per ask.
4. The unit count per ask.
5. A hard stop, not a silent pad, when a track that should
   carry sound decodes missing, empty, or all zeros. Declared
   silence controls are exempt by name in the job list, never
   by accident.
6. A saved copy or a hash of each extracted track beside the
   results file, so a later audit can check the session's own
   bytes instead of a source folder's.
