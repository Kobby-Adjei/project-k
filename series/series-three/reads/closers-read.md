# The two closers, read (Nov 4, 2026)

One battery, run clean on the free lane (L4, exit 0, token
check passed): topcheck_results.csv (12 rows) and
wording_results.csv (1440 rows), read from the files here, not
from the run report.

## Part 1: the top check closes the index account

Pasting a clip's entry-36 state (the post-norm final state)
into another run's post-norm slot at the final position
returns the pasted clip's own generate baseline EXACTLY: max
difference 0.0000 across all 12 cells (4 self lines, 4 tone
into silence, 4 silence into tone). Nothing after that slot
can change the number, and empirically nothing does.

What this settles:
- The index account is closed at both ends: exactness verified
  at layers 0, 18, 28 (all-positions pastes, index check) and
  now at the top, where entry 36 is pasted into the only slot
  it truly belongs to.
- The state-36 probe readout (round 2 and step 5 read pitch,
  onset, and presence at state 36) rests on a verified state:
  entry 36 is exactly the vector the output head reads.
- The double-norm reading of every "layer 35" patch stands as
  the only account consistent with all of it.

## Part 2: the wording battery

The same 360 property tones, four new forms, scored from the
first-token logits. Accuracy by the emitted word, accuracy by
the normalized share at 0.5, and the AUC of the share:

- pitch_reversed ("Is the tone low-pitched?"; truth: low):
  word 0.625, share 0.650, AUC 0.6331. The model answers No
  269 times, Yes 91. Share range 0.411-0.527.
- onset_reversed ("Does the tone start after the first
  second?"; truth: late): word 0.500, share 0.500, AUC 0.9813.
  The model answers Yes on all 360 asks. Share range
  0.626-0.703, entirely above 0.5.
- pitch_choice ("high or low?"; scored on the high share):
  word 0.500, share 0.500, AUC 1.0000. The model answers
  "high" on all 360 asks. Share range 0.625-0.870, entirely
  above 0.5.
- onset_choice ("early or late?"; early share): word 0.500,
  share 0.500, AUC 0.9425. The model answers "early" on all
  360 asks. Share range 0.763-0.878, entirely above 0.5.

## What it means

The bias is not a yes/no quirk. In every form, the spoken
answer collapses to the affirmative or first-named option:
yes to the reversed timing question even when the tone starts
early, "high" to every pitch choice, "early" to every timing
choice. And in three of the four forms the underlying lean
still ranks the truth almost perfectly (AUC 0.9813, 1.0000,
0.9425): the whole share distribution sits on one side of the
0.5 line while preserving the order. The line is misplaced in
every wording tried, and the ranking survives every wording
but one. So ChatGPT's format-effect alternative is half right:
the form decides which option the model defaults to; it does
not explain the information away, because the margin carries
the property in form after form.

The one exception matters: reversed pitch. Asked about
"low-pitched", the margin separates at only 0.6331, against
1.0000 for the "high-pitched" direction on the same clips.
Wording can degrade the margin itself, not just the cut. The
pitch readout is asymmetric: the model represents the high
side of the contrast far more cleanly than the low side, or
the low wording taps it worse. Either way, claims about the
margin should name their wording, and the property patch's
primary readout uses the original directions (AUC 1.0000 and
0.9988).

Prediction scorecard (registered in the build note): the
reversed timing question mirrored the bias exactly, as
predicted. The forced choices did NOT score high by word or
by share (0.500, 0.500); they scored perfectly only in the
margin. The prediction that a direct choice would read the
lean cleanly was wrong about the spoken answer and right
about the lean. Recorded as a miss.

## Standing claims after the closers

- Threshold account, widened: on these tones, the model's
  answer defaults to the affirmative/first option across
  question forms, while its margin ranks pitch and onset
  near-perfectly (original wordings 1.0000 / 0.9988; reversed
  onset 0.9813; choices 1.0000 / 0.9425; reversed pitch the
  exception at 0.6331). The wrong answers are produced at the
  decision cut, in every form tried.
- The index account is closed empirically at the top as well
  as at 0, 18, 28.
- Next, in the agreed order: the property patch (readout: the
  original questions' margins, both readouts kept), then
  MiniCPM.
