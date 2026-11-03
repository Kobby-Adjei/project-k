# Series Two, OLA (7B) diagnostics: what the run showed

Run Nov 2, 2026, on kobbytti@gmail.com Colab (L4, High-RAM).
Model: THUdyh/Ola-7b at pinned revision
d2fcaa533792480ce748321bea5101fed29d857e, authors' code at
commit f681820b32db398e62d430c4e5b4f3dd962bbed4, loaded in
bf16 form on rung 1, 18.2 GB on chip. Sanity: "Ready."
Battery: 87 asks, all answered, no hard stop, no run problem.
Results: series2_ola_results.csv in the kobbytti
project-k-clips folder (162,847 bytes). Feed proof printed
per ask from OLA's own path (decord frames and moviepy audio,
in process; duration, track level, nonzero share recorded per
row). Yes scores ARE exposed by this model's generate path
and are recorded on the yes or no asks: the probability of
the yes token at the first answer position, from the same
greedy call. First model in the project where that number
exists.

## The short read

OLA is the mirror image of MiniCPM. MiniCPM hears more than
it says. OLA says more than it hears. Fed pure silence, it
describes rooms, people, footsteps, and speech that are not
there. Asked yes or no, it says yes to the vacuum, the
sizzle, the beeps, and the water even when the clip is muted,
so its yes does not track the track. Underneath the yes, the
recorded numbers do move the right way for most sounds:
sound present scores higher than the same question on the
muted clip. The signal is graded; the spoken answer is not.

## The registered prediction, scored

Claude's prediction, written in plan.md before this run: if
the ear is the bottleneck, OLA should name the vacuum and
the sizzle where the speech ear only models did not, because
OLA also hears through BEATs, an ear built for sounds.

The letter of it came true. On the sound on clips, OLA names
both: vacuum yes at 0.820, sizzle yes at 0.789, the strongest
yes scores in their rows, and beeps and water besides.

The spirit of it failed. The muted control, the same silence
file under every question, also gets a yes for the vacuum
(0.673), the sizzle (0.746), the beeps (0.615), and the
water (0.587). A model that names the vacuum in silence has
not shown it heard the vacuum in sound. The inference the
prediction was built to test, that the sound ear is what the
other models were missing, does not survive its own control.
For OLA the bottleneck is not the ear. It is a yes that
fires too easily, and, in the open form, a habit of
inventing what is not there.

## Test 2: sound alone vs sound with picture, three question forms (48 asks)

Speech:
- Sound alone: open heard "The woman is speaking", presence
  Yes (0.612), transcription word perfect ("The blue kettle
  boils at seven forty two"). Speech is acknowledged here,
  unlike MiniCPM, which transcribed it and denied it in the
  same run.
- With the still: open heard "boiling sounds" and seen
  invented "a person in a kitchen with a pot on the stove".
  The picture in the file is a white circle on black. The
  model reported the sound the sentence is ABOUT (a boiling
  kettle) instead of the sound that is there (a voice), and
  built the matching kitchen from the words. Transcription
  in this arm: "The blue kettle boils at 742", near perfect.

Silence (true digital zeros):
- Open, both arms: full invention. Sound alone produced a
  bedroom scene with two people and "heard" speech and
  background noise. With the still, a different invented
  room, footsteps, and occasional speech.
- Presence: Yes (0.724 sound alone, 0.702 with still).
- Detail: "No, there is silence in this clip" (0.101) and
  "No, there is no sound at all in this clip" (0.129). The
  detail ask gets the truth the other two forms miss.

Water: caught everywhere, both arms, all forms, presence
0.883 to 0.907. The robust hit, as with every model so far.

Beeps:
- Presence and detail catch them in both arms (Yes 0.868
  and 0.915, detail "Beeps").
- The open form misses them in both arms and substitutes
  invented sounds: footsteps and wind (sound alone), a car
  and distant talk (with still). Same shape as MiniCPM's
  beeps: direct forms catch, the open form drops, except
  OLA fills the hole with invention where MiniCPM wrote
  "none".

## Test 3: yes or no, three versions, with yes scores (30 asks)

Versions: sound on with the still, muted with the still,
sound on with a plain gray frame. Muted is the control.

- Every matching question on a sound on clip: Yes, with the
  row's highest scores (speech 0.667, beeps 0.889, water
  0.883, vacuum 0.820, sizzle 0.789, mixed speech 0.643 and
  mixed vacuum 0.521).
- Every matching question on a MUTED clip: also Yes
  (beeps 0.615, water 0.587, vacuum 0.673, sizzle 0.746),
  except speech, which flips to No at 0.495, the one clean
  flip in the table.
- Lures partly work: the speech clip asked about water says
  No (0.370); the water clip asked about speech says No
  (0.147). But the muted speech file asked about water says
  Yes (0.587).
- Silence asked about silence: Yes (0.640 still, 0.718
  gray), which is the one muted-row yes that is correct.

Read: the binary answer is near useless for this model; it
says yes to the target sound whether the track carries it
or not. The yes score underneath is better behaved: for
speech, beeps, water, and vacuum, sound on scores clearly
above the muted control (gaps of about 0.17, 0.27, 0.30,
0.15). For sizzle the gap nearly vanishes (0.789 against
0.746). Claude's reason for wanting the number is proved
out in his own test: the number separates what the word
cannot. But separation in the score is not separation in
the answer, and any future use of this model family would
have to read scores, not words.

## Test 4: one tone, moved (9 asks)

- Presence: Yes all three times (0.756, 0.758, 0.739). The
  tone is heard.
- Early or late half: "second half" all three times. Wrong
  for 2, borderline for 6, right for 10. Identical answers
  to MiniCPM's on this form.
- Number: 2 became 1 (-1), 6 became 1 (-5), 10 became 3
  (-7). The numbers collapse toward the start of the clip.
Read: detection without a clock, again, but the lean is
opposite to MiniCPM's (which answered 4, 5, 12: errors +2,
-1, +2). OLA's timing errors are the largest yet measured
on this ladder.

## Where this leaves the stages (hypotheses, not proof)

- Feed integrity: settled for these asks, proof per ask.
- Event recognition: water robust; beeps recognized under
  direct forms; speech recognized and acknowledged (unlike
  MiniCPM); vacuum and sizzle produce the row's strongest
  yes scores when present, but the same yes fires in their
  absence, so recognition cannot be credited from the
  answers alone.
- Reporting: OLA's loss runs the other way from MiniCPM's.
  Where MiniCPM drops real sound from the report, OLA adds
  absent sound to it: invented scenes and sounds on pure
  silence in the open form, yes on muted controls in the
  forced form. The taxonomy's separate hallucination track
  is where most of OLA's failures live.
- Cross-modal interference: one vivid cell. Speech with
  the still came back as "boiling sounds" with an invented
  kitchen: the sentence's meaning steered both the heard
  and the seen fields. Small in count, large in kind.
- Timing: heard always, placed worst of the three models
  timed so far.

## The two-model Series Two picture

MiniCPM: the sound gets in and the report drops it. Speech
transcribed perfectly and denied in the same breath; vacuum
owned only under a forced choice whose muted control
flipped correctly.

OLA: the report adds what the sound never carried. Silence
described as rooms and voices; muted clips confirmed as
vacuum and sizzle; a graded yes score underneath that
moves the right way while the spoken yes does not.

Different designs lose, and invent, sound in different
places. That was Claude's Series Two framing on Nov 2, and
both runs landed inside it.

## Run notes

- The deepspeed install break (missing pkg_resources under
  setuptools 81 and up, building without isolation) was
  fixed before this run by pinning setuptools below 81 into
  the environment just before the deepspeed pass. Step 1
  passed in full on this run, deepspeed import OK. The fix
  is in both OLA notebook builders.
- One interim browser report during this run quoted the
  Colab session's own versions as Step 1's proof; the final
  report corrected itself against the saved notebook output.
  The proof of record is the notebook's printed block:
  torch 2.1.2+cu121, transformers 4.43.4 from the venv path,
  proof passed.
