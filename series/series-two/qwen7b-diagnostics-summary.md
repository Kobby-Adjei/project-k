# Series Two diagnostics: Qwen2.5-Omni-7B read

Run: Nov 2, 2026, kobbytti Colab, L4 High-RAM. Loaded the full
model Qwen/Qwen2.5-Omni-7B in fp16 (rung 1 of its ladder),
transformers 4.52.4. 87 of 87 asks, no hard stop. CSV
series2_qwen7b_results.csv (198,147 bytes) in the kobbytti
project-k-clips folder, read directly from the saved file.
Its Step 3 check passed; the exact SANITY RAW OUTPUT line was
never transcribed (the notebook output was folded and the
session limit blocked the re-read). The load log confirms the
fp16 full-model load in check and battery modes.

## Test 2, the two by two

Speech (the spoken sentence, loud at seconds 2 to 4):
- Open form: heard "none" in both arms.
- Presence form: "No." in both arms. Yes scores 0.2385 sound
  alone, 0.1093 with the still.
- Transcription form: word perfect in both arms ("the blue
  kettle boils at seven forty-two" sound alone; "The blue
  kettle boils at 7:42." with the still).
The ear has the speech; the report denies it exists. This is
MiniCPM's pattern exactly: loss in the reporting, not the ear.

Silence:
- Open correct ("none") both arms; name form correct ("No.")
  both arms.
- Presence: "No." sound alone, but "yes" with the still, a
  false alarm at score 0.1666.

Water:
- Open form sound alone MISNAMES it: "The sound of a bicycle
  tire rolling on the ground." With the still it names it
  right: "water trickling and a faint, rhythmic sound".
  Start 0 in both.
- Presence: Yes in both arms (0.6301, 0.4685).
- Name form ("What sound, if any... a short sound name or
  none"): "none" in BOTH arms, while presence said Yes to the
  same files. Asked to name the sound, it says there is none;
  asked yes or no, it says yes.

Beeps:
- Open form: "beeping" in both arms (start null sound alone,
  4.0 with the still; the beeps occupy seconds 3 to 6).
- Presence: Yes both arms (0.4772, 0.5877).
- Name form: "none" in both arms, same split as water.

## Test 3, yes or no with scores

Matching questions on sound-on clips: Yes for beeps (0.5552),
water (0.4685), vacuum (0.3370), sizzle (0.5250), and Yes on
the gray versions too. Muted controls: No for beeps, vacuum,
sizzle, but water FALSE ALARMS: the muted water file asked
about water answers "yes" (0.2334), and the muted speech file
asked the water lure also answers "yes" at the same 0.2334.
Speech is denied on its own clip in every version (still
0.0864, muted 0.0303, gray 0.2216, all "No"). In the mixed
file, speech is caught only on the gray frame (Yes, 0.4640);
the vacuum in the mixed file is never caught. Lures otherwise
behave. The silence question ("is it silence") answers Yes on
the silent file in all versions, which is correct there.

## Test 4, tone timing

Presence: Yes three times (0.8194, 0.7379, 0.7991). Early or
late: "second half" three times (wrong at 2 s, borderline at
6 s, right at 10 s). Numbers: 2 answered 5 (+3), 6 answered
10 (+4), 10 answered 10 (exact). Detection without a clock,
leaning late, the largest late errors so far.

## Stage read (hypotheses)

Feed integrity settled by the per-ask proof. Recognition:
real for water, beeps, vacuum, sizzle under forced choice.
Reporting is the biggest loss two ways: speech is transcribed
but never reported as a sound, and the name form collapses to
"none" for sounds the presence form confirms. Cross-modal
interference is small except the silence false alarm with
the still. Timing: detects, misplaces late.
