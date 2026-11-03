# Series Two, MiniCPM-o 2.6 diagnostics: what the run showed

Run Nov 2, 2026, on kobbytti@gmail.com Colab (L4, High-RAM).
Model: openbmb/MiniCPM-o-2_6, bf16, rung 1, pinned revision
06849bfd36da94da1f5a88fa17e8ba63f08a4c4d. Battery: 87 asks, all
answered, no hard stop, no run problem. Results:
series2_minicpm_results.csv in the kobbytti project-k-clips
folder (97 KB). Feed proof printed inside the session for every
ask; each extracted track hashed and saved. The yes score is
not exposed by this model's chat path, so no probabilities
exist for it; that was settled from the code before the run.

## The short read

The sound was reaching the model. The proof printed per ask
closes the feed question for these files: the Series One
misses cannot be blamed on a silent or padded track. What
changes the answer is the question form. The same sound, fed
the same proved way, is caught by one form and dropped by
another. The biggest loss sits in the telling, not the
hearing (ChatGPT's reporting stage). Timing is a separate,
smaller failure: the model hears the tone every time and
still cannot say when.

## Test 2: sound alone vs sound with picture, three question forms (48 asks)

Question forms: open (the repaired JSON form from Series One),
presence (a straight "is anyone speaking / is there a sound of
this kind" ask), detail (say what was said / name the sound).

Speech (the sentence "The blue kettle boils at 7:42"):
- Open, both arms: heard "none". Sound alone arm also put
  "a kettle boiling" under seen, which is the sentence's
  content leaking into the picture slot.
- Presence, both arms: "No." Speech denied outright.
- Detail, both arms: "The blue kettle boils at 7:42."
  Word perfect, both arms, both repeats.
Read: the speech reaches the model whole. Asked to write it
down, it writes it down. Asked whether speech is there, it
says no. Asked openly what it heard, it says none. The loss
is in the report, not the ear, and the picture changes
nothing.

Silence (true digital zeros):
- Open: heard "none". Correct.
- Presence: "Yes" in both arms. A false alarm on nothing.
- Detail: "No."
Read: the presence form is unreliable in both directions:
No on real speech, Yes on pure silence.

Water:
- Every form, both arms: caught. "water running" (sound
  alone), "water dripping" (with picture), presence Yes,
  detail "Running water". The robust hit, as in Series One.

Beeps:
- Sound alone: open says "none", but presence says Yes and
  detail says "Beep". The beeps get in; the open form drops
  them.
- With picture: open says "beeping sound", presence Yes,
  detail "Beep". Full hit.
Note against Series One: there, the open form on the same
beeps file said "none" with the picture feed. One open-form
cell flipped between runs. Greedy decoding should repeat, so
treat the open form on beeps as near a threshold: small
shakes flip it. The finding is the fragility, not either
single answer.

Arm comparison overall: sound alone vs sound with picture
barely matters. Speech is denied in both, water caught in
both. The picture is not what saves or kills a sound here.

## Test 3: yes or no, three versions of each file (30 asks)

Versions: sound on with the swap still, muted with the same
still, sound on with a plain gray frame. The muted version is
the control: if the answer follows the sound, it must flip
to No when the track is removed.

- Beeps: Yes, No, Yes. Correct pattern. The answer tracks
  the track.
- Vacuum: Yes, No, Yes. Correct pattern. FIRST TIME this
  model has owned the vacuum in any form. Under a forced
  yes or no, with the muted control flipping as it should,
  the vacuum registers. In the open form it still says none.
- Sizzle: Yes, No, No. Sound on vs muted is the correct
  pair; the gray version breaks it.
- Water: repeats split (Yes/No on sound on, No/No muted,
  Yes/No on gray). Unstable.
- Mixed (speech over vacuum): repeats split (No/Yes on
  sound on, No/No muted, No/Yes on gray). Unstable.
- Speech: No on every version, including sound on. Matches
  Test 2: speech is never acknowledged as a sound.
- Silence: Yes on every version, including muted (which is
  the same silence). The standing false alarm from Test 2.

Read: the yes or no answers partly track the track (muted
flips beeps, vacuum, and sizzle to No), which is real
evidence those sounds reach the model and can be reported
under a forced choice. But the instrument wobbles: repeats
disagree on water and mixed, and silence draws a constant
Yes. A yes or no from this model is a weak witness, usable
only with its muted control beside it.

## Test 4: one tone, moved (9 asks)

1000 Hz tone, half a second, placed at 2, 6, and 10 seconds
in a 12 second clip, onsets measured exact in the build.
- Presence: Yes, Yes, Yes. The tone is always heard.
- Early or late half: "second half" all three times. Wrong
  for the 2 second tone, borderline for 6, right only for 10.
- Number: 2 became 4 (+2), 6 became 5 (-1), 10 became 12
  (+2, the very end of the clip).
Read: detection without a clock. It knows a tone happened
and roughly guesses late. Three tones, one ask each: a
small cell, stated as such.

## Where this leaves the stages (hypotheses, not proof)

- Feed integrity: settled for these asks. Proof printed per
  ask, tracks hashed, no silent padding anywhere in the run.
- Event recognition: water robust; beeps and vacuum
  recognized under direct or forced forms; sizzle fragile;
  speech recognized as language (perfect transcript) but
  never as a sound event.
- Reporting: the largest measured loss for this model.
  Speech is the clean case: transcript perfect, presence
  denied, open report "none", in both arms, with the feed
  proved. Beeps in the sound-alone arm show the same shape
  in miniature.
- Cross-modal interference: small here. The arms barely
  differ, and no bed was needed to bend a report this time.
  The bed cells were not rerun in Series Two; the Series One
  bed result for Pegasus still stands alone.
- Timing: detection yes, clock no. Errors +2, -1, +2 on a
  2, 6, 10 ladder.

## Run notes

- Three swap files (silence, vacuum, mixed) were missing
  from both Drive clips folders; only 4 of 7 swap files had
  survived the account migration. The local originals were
  restored to the kobbytti folder before Step 2 passed.
  All 7 swap files, 7 gray versions, and 3 tone files were
  then built and confirmed in session.
- The stuck passkey session was abandoned. A fresh session
  drew the phone prompt instead (Gmail app, Yes, 28) and
  completed. Session luck, recorded so the next run expects
  it.
- Sanity: the notebook check gate passed (exit 0, rung 1).
  The runner's literal sanity text was not visible in the
  cell output of the check run; the gate result is what is
  recorded, no quote invented.
