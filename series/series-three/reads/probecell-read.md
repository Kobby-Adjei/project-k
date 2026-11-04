# The probe cell: read (Nov 4, 2026)

Run Nov 4, 2026, free lane (kadjei1), L4, exit 0, token
check passed. Files: `probecell_probe.csv` (2400 rows),
`probecell_lean.csv` (32 rows), `probecell_run_log.txt`.
Design and registered predictions: `probecell-build-note.md`.
All numbers below recomputed from the CSVs by Muse; where
the run's printed readout pools things misleadingly, this
read separates them and says so.

## The instruments, honestly

The probe trained in this run is WEAKER than the graded
battery's: unpatched validity MAE 0.2117 octaves at the
audio positions (states 5-36), 0.2443 at the final position,
against the graded battery's 0.05-0.09. Registered
prediction 1 (validity <= ~0.10) FAILED. The likely reasons
are the smaller training set (96 clips) and the silence
background (no shared background scheme); stated, not
chased, because the cell's separations (0.46 and 1.17
octaves between pair members) are still 2-5x the probe's
error, so the cell resolves. Any finer claim would need
the stronger probe.

Paste-site check, separated by position (the printed
readout pooled them into one misleading 0.4730): at the
AUDIO positions the pasted state reads at probe accuracy
(mean err vs donor 0.2482, the validity level) — the paste
lands. At the FINAL position the paste-site row is the
recipient's own (computed from pre-paste inputs), err
0.6978 — expected by construction, not a failure.

## The cell's answer

Downstream of the paste (states layer+2 to 36), the probe
at the AUDIO positions follows the DONOR's exact frequency:

- Low pairs (400 vs 900, separation 1.17 oct): closer to
  the donor in 100% of rows; mean err vs donor 0.084
  octaves, vs recipient 1.106. The probe recovers the
  donor's frequency essentially exactly, at every paste
  layer 14-20.
- High pairs (1600 vs 2200, separation 0.46 oct): closer
  to the donor in 82% of rows; err 0.286 vs 0.559.

At the FINAL position the following is partial: closer to
the donor in 69% (low) / 72% (high) of downstream rows,
errors 0.528/0.640 and 0.205/0.374.

So the pasted state carries the fine pitch into the
downstream computation at the sound's own positions, in
full, while — from the fixpack — jitter-scale pitch
differences move the yes/no lean not at all in the same
layers. The coarseness is in the lean's reading of pitch,
not in the state's carrying of it. At the answer position
the fine pitch arrives weakened, which is consistent with
the lean reading a coarse summary of a fine state.

## The premise check, and a scoping correction it forces

The lean in THESE pairs did not sit still: the pairs'
baseline gaps average 0.060 share points, and the patched
share moves about two-thirds of the way to the donor
(mean |patched - recip| 0.0415 against |patched - donor|
0.0198). These pairs are wide (400 vs 900 Hz), and the
model's own high/low boundary evidently sits inside that
span, so the lean partially separates them. Combined with
the regression cell, the scoped claim is now: the lean is
blind to pitch differences at the jitter scale (the
property tones' +-2%; corr ~0 within band, fixpack
within-band leans unmoved) and partially sighted at the
octave scale. The earlier unscoped sentence "the lean
carries no within-band pitch signal" is corrected to that.
The probe result is unaffected: at the audio positions the
state carries the exact frequency regardless of which
scale the lean can see.

## Prediction scorecard

1. Validity <= 0.10 octaves: FAILED (0.2117). The cell
   still resolves on separation, stated above.
2. Paste site reads donor: at the audio positions, yes,
   at probe accuracy; the pooled printed line (0.4730)
   mixed in the final position, where the paste-site row
   is the recipient's own by construction.
3. Downstream closer-to-donor: YES at the audio positions
   (100% / 82%), partial at the final position (69% /
   72%). The "state carries fine pitch; the lean reads it
   coarsely" branch is the result, with the lean's
   scale-scoping above.

## Correction note (Nov 4, 2026, the council's third audit)

Three fixes. (1) The audio-position conclusion is stronger
than stated: 98.6% of the low pairs' downstream rows land
within the probe's own validity error of the donor — the
following is donor-matched to the instrument's limit, not
merely resolvable on separation. (2) The high pairs' final-
position figure (72%) is UNINTERPRETABLE in principle:
their half-separation (0.225 octaves) is smaller than the
probe's validity error at the final position (0.2443), so
the instrument cannot separate those frequencies there. It
is dropped as a number. (3) "Arrives weakened" is retired.
The final position's following is a RAMP with downstream
distance (closer-to-donor rate, low pairs: 38.8% just after
the paste, 74.0% mid, 86.5% late; high pairs: 35.0 / 75.0 /
96.2) — fine pitch reaches the final position late, which
is what a relay through the intervening positions predicts.
As an effect size the pooled rates were also backwards
(normalized following 0.096 low vs 0.376 high); the ramp is
the reportable form. Separately, on the lean: the
octave-scale pairs moved 69.9% of their gap toward the
donor in 31 of 32 rows — near-unanimous direction on a
small dynamic range; "partially sighted" stands with that
strength noted.
