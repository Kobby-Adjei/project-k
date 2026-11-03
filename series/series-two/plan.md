# Series Two: the intervention series (Project K)

Opened Nov 2, 2026, on Kobby's word: "Audit our feed first, then
OLA," and the council's suggested tests become a new body of work
called Series Two.

## What Series Two is

Series One showed where the sound goes missing, model by model.
Four models sat the same traps, and each one failed in its own
shape: Pegasus reported only what a speech transcript carries, the
small Qwen heard fragments and misfiled them, the big Qwen heard
the most and invented the most, and MiniCPM-o reported almost
nothing beyond water.

Series Two asks the next question: which stage loses the sound?
It does not ask it with bigger batteries. It asks it with
controlled changes, one at a time, on the same files. Change one
thing, watch what moves, and the stage that owns the failure
starts to show itself.

The working frame is ChatGPT's taxonomy from the council replies
of Nov 2, 2026. These are stages to test, not proven labels:

1. **Feed integrity.** Did the sound actually reach the model,
   in the form the model was supposed to receive it?
2. **Event recognition.** Given the sound arrived, did anything
   inside register the event at all?
3. **Cross-modal interference.** Does the picture change what
   the sound alone would have produced?
4. **Reporting.** Was the event registered but never written
   into the answer?
5. **Timing.** Was the event reported but never placed in time?

Hallucination, sounds reported that were never there, is tracked
alongside all five, not as a stage of its own.

## The test list, in run order

**Test 1: the feed audit.** Status: DONE, Nov 2, 2026. Done
first, locally, no GPU. The
MiniCPM runner's audio path is read line by line, and the 12
battery clips are measured as decoded files: audio stream,
duration, level, and a per-second trace. It answers the council's
sharpest charge, that the MiniCPM run never proved sound reached
the model per row. Record: pilot2/minicpm-o/feed-audit.md.

**Test 2: the 2x2 on MiniCPM.** Status: built into the Series
Two diagnostic notebook, Nov 2, 2026 (Drive id
1-ZvXPHCv1csynqY9V6x5YOxs5HtlVNVh); not yet run. Sound alone
versus the same sound
with a still picture, crossed with two questions: the open
description, and the direct one (is speech present; write down
the words). The speech-alone file goes first as the baseline,
true silence rides along as the negative control, then the water
and beep files take the same four cells. This is ChatGPT's pick
for the most information per run: it separates a broken feed, a
picture that interferes, and a question that suppresses, from
each other.

**Test 3: yes/no scoring on the swap files.** Status: built into
the same diagnostic notebook, Nov 2, 2026; not yet run. One
finding landed at build time: MiniCPM's chat path exposes no
score for the yes token (its decode path never requests output
scores), so the yes-score column reads "not exposed" and the
words are the record. Claude's pick. Ask
"Is there the sound of water / a vacuum cleaner / sizzling /
beeping / a person speaking in this clip? Answer yes or no," on
the existing swap files in three versions each: sound on, sound
muted, and sound under a plain gray frame. Where an open model's
ask path exposes scores, read how strongly the model leans yes;
where it does not, the plain yes/no words are the record. The
full swap set runs this time, vacuum and silence included, which
closes the gap the audit proved: no open model has ever been
asked about a vacuum soundtrack or about true silence. The
reading is fixed in advance: if yes does not move between sound
on and muted, the sound never got in; if yes rises but the free
answer still says none, the loss is at the reporting stage; if
yes stays high on a muted kitchen picture, the scene is guessing.

**Test 4: the tone-shift timing control.** Status: built into
the same diagnostic notebook, Nov 2, 2026; the three tone files
are built and measured (series-two-2026-11-02/tone/), and the
notebook's Step 2 rebuilds them in session and saves them into
the clips folder; not yet run. One verified tone,
placed at about 2, 6, and 10 seconds. Presence, early-or-late
judgment, and the numeric start are asked as three separate
questions. Timing is scored only on the trials where the sound
was detected at all, so a timing failure can never again hide
inside a hearing failure, or the other way around.

**Test 5: OLA rides the same cells.** Status: the OLA
notebook's build fix has landed (Nov 2, 2026); its run waits
until the MiniCPM diagnostic notebook above has run, per
Kobby's order. Once it runs, OLA takes tests 2 to 4 in the same
shapes, on
the same files. Per Kobby's order, its run comes after the feed
audit, and the diagnostic cells come before any wider battery on
it.

## The pre-registered OLA prediction

Written down on Nov 2, 2026, before OLA has run a single ask, at
Claude's prompting:

If the ear is the bottleneck, OLA should name the vacuum and the
sizzle where the speech-ear-only models did not. OLA hears
through two ears: Whisper large-v3 for speech, and BEATs, an ear
built for sounds and music. The models that missed the vacuum
and the sizzle hear through a speech ear alone. If OLA also
misses them, the bottleneck is probably not the ear, and the
reporting and interference stages move to the front of the line.
If OLA names them, the ear moves to the front. Either way the
prediction was written before the run, and the run cannot be
allowed to rewrite it.

## Rules carried over from Series One

- Same files, same questions inside a comparison. A difference
  in answers must never be explainable by a difference in inputs.
- Raw answers are always saved, in full.
- Failures print loudly and are never folded into empty answers.
  An error is recorded as an error, never as silence.
- Every claim is capped at the door each model was actually fed:
  that model, through that feed, on those files, on that date.
- Feed differences are written beside the results, never hidden:
  frame rate, picture size, how the sound was cut and carried.
- Nothing here is a ranking. Each test is a behavior report on
  one model through one door, and the stages above stay
  hypotheses until an intervention separates them.
- New from the audit: any future run prints its feed proof from
  inside the session, per ask, from the model's own feed path.
  What those prints must contain is listed at the end of
  pilot2/minicpm-o/feed-audit.md.

## Run record: MiniCPM diagnostics (Nov 2, 2026)

DONE. 87 of 87 asks answered on the fresh run (kobbytti, L4);
series2_minicpm_results.csv saved in the clips folder. Feed
proof printed per ask; the feed question is closed for these
files. Headlines: speech transcribed word perfect under the
detail form and denied under the open and presence forms in
both arms (reporting stage); vacuum owned for the first time
under forced yes or no with the muted control flipping to No;
yes or no answers wobble on repeats and false-alarm on pure
silence; tone timing errors +2, -1, +2 on the 2, 6, 10
ladder. Full read: minicpm-diagnostics-summary.md in this
folder. Next per the run order: OLA rides the same cells,
with Claude's prediction already registered above.

## Run record: OLA diagnostics (Nov 2, 2026)

DONE. 87 of 87 asks answered (kobbytti, L4, bf16 rung 1,
18.2 GB); series2_ola_results.csv saved in the clips folder.
Full read: ola-diagnostics-summary.md in this folder.
Prediction scored: the letter held (OLA names the vacuum at
0.820 and the sizzle at 0.789 on the sound on clips, the
strongest yes scores in their rows) and the spirit failed
(the muted control draws Yes for both, 0.673 and 0.746, so
naming does not prove hearing; the ear bottleneck inference
does not survive its control). OLA's failure shape is the
mirror of MiniCPM's: invention on silence, yes on muted,
while the yes scores under the words move the right way for
speech, beeps, water, and vacuum. Timing errors -1, -5, -7,
the largest yet on the ladder. Next on the running order:
Baichuan-Omni 1.5.

## Correction and order ruling (Nov 2, 2026)

Kobby caught a narrowing: Claude's yes/no test was proposed
for the open models already on the table (Qwen 7B, Qwen 3B,
MiniCPM) with OLA riding along, and Pegasus by text answers
plus lures. The plan as executed ran MiniCPM and OLA only.
His ruling on Nov 2, 2026: the Qwens rerun through the
Series Two cells next, before Baichuan-Omni. Pegasus's text
version of the yes/no test stays owed after that.

## Run record: Qwen2.5-Omni-7B (Nov 2, 2026, completed about 09:31 AM EST)

Full model in fp16 (rung 1), kobbytti Colab, L4. 87 of 87
asks, CSV series2_qwen7b_results.csv (198,147 bytes) in the
clips folder. Step 3 check passed; the exact SANITY RAW
OUTPUT line was never transcribed (folded output plus a
session-limit block on the re-read); the load log confirms
the fp16 load. Read: qwen7b-diagnostics-summary.md. Headline:
speech transcribed word perfect in both arms and denied as a
sound in both (MiniCPM's reporting loss exactly); the name
form answers "none" for water and beeps that the presence
form confirms; water misnamed sound alone, named right with
the still; Test 3 mostly clean with a muted-water false yes;
tone numbers late (+3, +4, 0).

## Run record: Qwen2.5-Omni-3B (Nov 2, 2026, completed about 10:03 AM EST)

Fp16, kobbytti Colab, L4. Sanity "I'm ready." 87 of 87 asks,
CSV series2_qwen_results.csv (197,624 bytes); two independent
runs wrote the identical size. Read:
qwen3b-diagnostics-summary.md. Headline: the cleanest Test 3
pattern of the four models (yes tracks the track on every
non-speech sound); the still picture wrecks its speech read
(presence Yes alone, No with still; transcription collapses
to "742"), the strongest cross-modal interference measured;
presence false alarms on silence sound alone; tone presence
fails at 6 and 10 s and its numbers collapse early
(-1, -4, -7.9), OLA's shape.

## Series Two status after the Qwen reruns

All four open models plus the council tests are done:
MiniCPM, OLA, Qwen 7B, Qwen 3B through the same 87 cells.
Still owed from Claude's Q4: Pegasus's text version (yes/no
answers plus lures through the Twelve Labs door; no scores
exist there). Next new model per NEXT-MODELS: Baichuan-Omni
1.5, its Series Two build not yet started.

## Equal-tests ruling (Nov 2, 2026, Kobby)

Every model goes through the same Series One battery before
any post speaks across models: the real clips clean, muted,
and blacked out under the one repaired question, the swap
matrix, and the matched bed test with a baseline that model
actually passes. Owed cells: OLA owes the whole Series One
battery (Series Two cells were its only run); the 3B owes
the proper swap cells, the bed, the blackout, and the muted
pairs; the 7B and MiniCPM owe the blackout, a valid bed, and
the muted pairs. Pegasus is complete. Only then does the
LinkedIn series open with the whole board.

## Lane note (Nov 2, 2026)

The equal-tests ruling for completing the Series One
battery lives in series-one-2026-11-01/EQUAL-TESTS.md, not
here, at Kobby's direction: it must not conflict with the
Pegasus Series Two work in this plan. Nothing in Series Two
changes because of it.

## Equal battery run records (Nov 2, 2026)

- Qwen2.5-Omni-7B, EQUAL battery, COMPLETE Nov 2, 2026
(run 11:36 AM to 11:50 AM EST, harness-measured): kobbytti
Colab, L4 High-RAM, fp16, transformers 4.52.4. The fresh
session drew the phone prompt (Yes + 8), completed by
Kobby. Step 2 READY line exact. Step 3 sanity answer
"One", check passed. Step 4: battery exit code 0, every
ask answered, every feed proved; equal_qwen7b_results.csv
listed at 101.44K in the Colab Files pane (modified
11:48 AM EST) and saved to the clips folder. READ DONE
Nov 2, 2026: downloaded by Drive id
1VDtyUg4pUm91cmy8IkzjlBbD5AkOWghg (the share's address
bar URL), 103,873 bytes, 41 data rows, read here. Read:
qwen7b-equal-diagnostics-summary.md. Headline: vacuum
missed and replaced with cooking sounds from the eyes;
blackout turns the vacuum into an invented door, the
same label the 3B gave this clip; water and eggs sizzle
caught with the picture on, lost in blackout; speech
denied in every form, and its plain-question baseline
on clean speech answers "silent.", which VOIDS its bed
rows through that question; the transcription door
answers "no words" on all three matched files.
- Qwen2.5-Omni-3B, EQUAL battery: launched Nov 2, 2026
~11:51 AM EST, after the 7B equal checkpoint. MiniCPM
and OLA equal runs queue behind it, one at a time.
- Qwen2.5-Omni-3B, EQUAL battery, COMPLETE Nov 2, 2026
(run 11:50 AM to 12:09 PM EST, harness-measured):
kobbytti Colab, L4 High-RAM, transformers 4.52.4. No
verification prompt; Drive mount cleared by account
chooser. Session handling: terminated only the finished
7B equal copy. Step 2 READY line exact; Step 3 check
passed; Step 4 exit 0, every ask answered, every feed
proved; equal_qwen_results.csv verified in the clips
folder at 102,233 bytes. Builder note: this notebook's
Step 1 text says the chip must be a T4; it ran on L4 and
the stack proof passed, so the note is stale text, not a
requirement. Read of the CSV owed; share to kadjei1 in
flight. Note: the kadjei1 connector still cannot see the
7B equal CSV ~15 min after its share (indexing lag); the
fallback is the file's Drive URL from the browser, then a
download by id.
- MiniCPM-o 2.6, EQUAL battery: launched Nov 2, 2026
~12:10 PM EST, after the 3B equal checkpoint. OLA equal
queues behind it.
- MiniCPM-o 2.6, EQUAL battery, COMPLETE Nov 2, 2026
(run 12:09 PM to 12:32 PM EST, harness-measured):
kobbytti Colab, L4 High-RAM, bf16, transformers 4.44.2
(its own pin). No verification prompt. Session handling:
terminated only a finished 7B equal copy. Sanity answer
"A cat sits on a mat." exact. Step 4: 41/41,
equal_minicpm_results.csv saved to the clips folder
(64 KB, modified 12:27 PM EST). Read owed; share to
kadjei1 in flight.
- OLA-7B, EQUAL battery: launched Nov 2, 2026 ~12:33 PM
EST, after the MiniCPM equal checkpoint. Last of the
four equal runs.
- Qwen2.5-Omni-3B, EQUAL battery, READ DONE Nov 2, 2026:
equal_qwen_results.csv (102,233 bytes, 41 rows) fetched
by browser download and read here. Read:
qwen3b-equal-diagnostics-summary.md. Headline: vacuum
clip answered "a door is opened and closed" again;
blackout makes it hear nothing on any real clip; swap
speech returns an empty shell and swap mixed collapses
into number garbage; both bed doors fail their baseline
("silent", "no words"), so its bed is void both ways.
- MiniCPM-o 2.6, EQUAL battery, READ DONE Nov 2, 2026:
equal_minicpm_results.csv (65,086 bytes, 41 rows)
fetched the same way and read here. Read:
minicpm-equal-diagnostics-summary.md. Headline: open
report drops vacuum, sizzle, mixed, and speech; muted
discipline perfect; blackout hears nothing; the plain
bed baseline fails ("The audio is silent."), but the
transcription door returns the whole sentence ("The
blue kettle boils at 742.", digits for "seven forty
two") identically on all three matched files, so its
bed reads clean through the door: neither bed bends
its words.
- OLA-7B, EQUAL battery, STOPPED at the Step 3 sanity
gate Nov 2, 2026 (run 12:32 PM to 12:48 PM EST,
harness-measured): the model loaded in bf16 (load log
confirms), transformers 4.43.4, but the check exited
code 5, the sanity gate. DIAGNOSIS (Nov 2, 2026, from
the notebook sources, not a report): a build bug in
the equal notebook. OLA's sanity gate asks "Reply with
the word ready." about a media clip,
/content/clips/probe_tone.mp4, but the equal Step 2
copied only the 13 battery files and dropped the probe
clip (the Series Two Step 2 carried it in its base
list). The ask therefore errored on a missing file
before the model could answer; the gate never tested
the model. The saved notebook outputs hold only the
667-char tail (Colab truncation), so the exact error
line is unrecoverable, but the missing file is proven
from the sources: the runner references probe_tone
exactly once (the sanity clip) and Step 2 never
brought it. FIX: build_equal_ola_notebook.py patched
so Step 2 also copies and requires probe_tone.mp4;
notebook rebuilt and verified (only the title and
Step 2 cells differ from the stopped run; runner and
battery byte-identical); the fixed notebook replaced
the Drive original (id 1WIm_eFa26-ssVf-e4d-K8U50yWG3hvlM).
RERUN launched Nov 2, 2026 ~1:20 PM EST. The rerun's
first two rolls (1:17 PM and 1:22 PM EST) both drew the
passkey-only screen at the Drive mount and were
abandoned untouched; the account is in its strict
phase again after today's many verification events.
PARKED for a verification cool-down at ~1:30 PM EST:
no more rolls until the account cools (earlier today
the same phase passed and runs began walking through
with no prompt at all). The fixed notebook is in
place; when the rerun goes, it is the same brief as
the other equal runs plus the SANITY RAW OUTPUT line.
