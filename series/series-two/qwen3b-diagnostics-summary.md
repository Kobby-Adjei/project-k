# Series Two diagnostics: Qwen2.5-Omni-3B read

Run: Nov 2, 2026, kobbytti Colab, L4 High-RAM. Loaded
Qwen/Qwen2.5-Omni-3B in fp16, transformers 4.52.4. Sanity
answer word for word: "I'm ready." 87 of 87 asks, no hard
stop. CSV series2_qwen_results.csv (197,624 bytes) in the
kobbytti project-k-clips folder, read directly from the saved
file. Two independent runs of this battery wrote the file at
the identical size; the read is from the final saved version.

## Test 2, the two by two

Speech (the spoken sentence, loud at seconds 2 to 4):
- Sound alone: the open form heard "boiling" with a start of
  2.14 s. It turned the speech into a wrong sound name but
  timed it near the true 2 s start. Presence said Yes
  (0.8229), the only model and form in Series Two where the
  presence question catches the speech on its own clip.
  Transcription near perfect: "the blue kettle boils at
  seven forty two".
- With the still, everything breaks: the open form returns
  empty answer shells, presence flips to "No" (0.2629), and
  the transcription collapses to "742". The picture does not
  just fail to help; it wrecks a speech read that worked
  sound alone. The strongest cross-modal interference
  measured in Series Two.

Silence:
- Open form correct sound alone ("no sound"); name form
  correct ("No.") both arms.
- Presence sound alone FALSE ALARMS: "Yes." at 0.7756 on a
  silent track. With the still it corrects to "No." (0.3777).

Water:
- Sound alone, the open form invents a whole scene: seen "a
  person is using a sewing machine to sew a piece of
  fabric", heard "the sound of the sewing machine humming
  and the occasional click of the needle" (on an audio-alone
  row with no picture at all). With the still it names the
  water right: "a sound of a water tap running".
- Presence Yes both arms (0.8519, 0.7678). Name form "none"
  in both arms, the same name-versus-yes split as the 7B.

Beeps:
- Open form sound alone: "none" (a miss). With the still:
  "a beep", start 2.41.
- Presence Yes both arms (0.6833, 0.8542). Name form "none"
  both arms.

## Test 3, yes or no with scores

The cleanest forced-choice pattern of any Series Two model.
Matching questions: Yes on sound-on, No on muted, for beeps
(0.9331 vs 0.2736), water (0.7678 vs 0.2235), vacuum (0.6427
vs 0.2836), sizzle (0.7221 vs 0.3435), and the gray versions
agree. Lures behave. Speech breaks the pattern by picture,
not by sound: denied on its own clip with the still (No,
0.3479) but caught on the gray frame (Yes, 0.5422); the
mixed file's speech is likewise caught only on gray (Yes,
0.7561), and its vacuum is never caught. The silence question
answers Yes on the silent file in all versions, correct
there.

## Test 4, tone timing

Presence FAILS at the longer onsets: Yes at 2 s (0.5010),
"No." at 6 s (0.4880) and "No." at 10 s (0.4222), the scores
sitting just under the line. Early or late: "second half"
three times. Numbers: 1.0 at 2 s (-1), 2.0 at 6 s (-4), 2.1
at 10 s (-7.9): an early collapse, the same shape as OLA's,
worst at the longest wait.

## Stage read (hypotheses)

Feed integrity settled by the per-ask proof. Recognition
under forced choice is the best measured: its yes answers
track the track on every non-speech sound. Its losses live
elsewhere: the picture breaking speech (interference), a
presence false alarm on silence sound alone (reporting),
invented scenes on the open form (hallucination track), and
a timing read that collapses early.
