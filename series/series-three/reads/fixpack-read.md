# The fixpack: read (Nov 4, 2026)

The council's corrections battery, run Nov 4, 2026, free lane
(kadjei1), L4, exit 0 both modes, token check passed, no errors.
Files: `fixpack_results.csv` (408 rows: Part A shams 264 + real
pairs at layer 21, 48 + mediation 96), `fixpack_wording.csv`
(216 rows), `fixpack_openform.csv` (54 rows), `fixpack_run_log.txt`.
Design and registered predictions: `fixpack-build-note.md`.
All numbers below recomputed from the CSVs by Muse; the run's own
printed readout matches the recomputation exactly.

## Part A: the gap-matched shams

The sham donors now match the recipient on the target property and
differ on the other property by a full gap (pitch sham: other gap
0.131; onset sham: other gap 0.116; target gaps 0.041 and 0.016).
Raw toward-donor moves of the lean, mean:

| sham read | L0 | 14-18 | 19-20 | 21+ |
|---|---|---|---|---|
| pitch sham, pitch read (gap 0.041) | +0.036 | -0.002 | +0.009 | -0.001 |
| pitch sham, onset read (gap 0.131) | +0.135 | +0.147 | +0.121 | +0.013 |
| onset sham, onset read (gap 0.016) | +0.018 | +0.019 | +0.015 | +0.001 |
| onset sham, pitch read (gap 0.116) | +0.116 | +0.074 | +0.061 | +0.012 |

What this says, carefully:

- The other property transplants in both arms: a same-pitch
  donor's onset moves the onset lean by its full gap through
  layer 20; a same-onset donor's pitch moves the pitch lean
  fully at layer 0 and at about 55-65% through layers 14-20.
- The target property splits by property. Onset: even the sham's
  tiny within-band gap (0.016) transplants at ~100% through
  layer 20. Pitch: the sham's within-band gap (0.041) transplants
  at layer 0 and then does NOT follow at layers 14-18 (raw move
  -0.002), while the real pairs (donor across the pitch band,
  gap 0.103) move the pitch lean 80-97% through layer 18.
- So the transplant is not simply wholesale, and not simply
  property-specific. In the mid-window, the pasted state's onset
  content governs the onset lean completely and the pitch lean
  partly; its pitch content governs the pitch lean when the
  pitch difference is a band difference, and not when it is a
  within-band difference. One alternative reading stands open:
  within-band pitch-lean differences may carry little pitch
  signal to follow (baseline jitter), while within-band onset
  differences demonstrably do carry signal (they transplant).
  n = 6 pairs per sham arm; the asymmetry is consistent across
  arms and layers, and it is stated at that strength only.

Layer 21, now measured for the real pairs: pitch fades smoothly
(58% at 19, 53% at 20, ~28% at 21, ~26% at 22); onset holds
through 20 (96-100%) and is already gone at 21 (14-18%). The
onset paste window therefore closes between layers 20 and 21,
one layer earlier than the morning's "21-22" wording; pitch's
fade spans 19-22.

## Part B: one captured state, six wordings

Percent of the pair's own gap, matched pairs (pitch forms on
pitch pairs, onset forms on onset pairs), layers 0 / 19 / 22:

- pitch original: 100 / 60 / 28. pitch reversed: 89 / 79 / 50.
  pitch choice: 102 / 60 / 43.
- onset original: 102 / 91 / 17. onset reversed: 89 / 57 / 23.
- onset choice: the margin is saturated at baseline (mean base
  share 0.809, donor 0.813, mean gap 0.021), so percentages are
  meaningless there; in raw shares the pasted value sits at the
  donor's at layer 0 and back at the recipient's by 19.

Cross reads move too (pitch form on onset pairs: 63 / 83 / 14;
onset form on pitch pairs: 125 / 74 / 11). Two conclusions:
(1) the transplant reaches every wording's margin in the same
window; no wording is exempt, including the reversed pitch
wording whose margin behaviorally reads onset. The fear that
"the paste moved a readout, not the state" is answered: the
paste moves the state, and each wording reads the moved state
in its own mixture. (2) Each wording's margin still responds
to both properties' pastes, so a margin remains a per-prompt
mixture, not a property meter.

## Part C: the mediation cells are VOID (my design error)

All 96 mediation rows are bit-identical to the matched
audio-only rows (max difference 0.0000). The reason is
structural, found while checking the file: the paste hook
rewrites the paste layer's OUTPUT, so at that layer the
post-block rows are computed from pre-paste inputs and are
already identical to the recipient's own. Restoring them
restores what is already there; the cell tested nothing.
The mediation the council specified needs the recipient's
post-block rows restored at every layer from the paste layer
upward, which is where those rows actually diverge. That
corrected cell is the retest; it has not run. Recorded as my
error, in the open, the same day.

## Part D: open-form controls

Unpatched recipients (the 6 low tones): 3 "silent" (the
early-band three) and 3 named answers (brief beep, busy signal,
telephone dial tone; the late-band three). Unpatched donors
(the 6 high tones): "beep", 6 of 6. Real pastes (high donor):
18/18 "beep" at layers 14/19/24 (run 1) and 6/6 "beep" at
layer 28 (this run); at layer 34 the answers dissolve into
the recipients' own variety (3 silent, 3 named). Sham pastes
(low donor, other onset) at the same layers: mostly not
"beep" (layer 14: 3 beep, 3 silent; layer 19: 5 silent;
layer 24: 4 silent; layer 28: 4 silent; layer 34: mixed).

So the 18/18 is rehabilitated in a narrower form: naming
follows the donor's band under a real paste, at layers 14
through 28, and the gap-matched sham at the same layers does
not produce it. It is a paste effect, not a battery artifact.
Two scopings stay: the donors say "beep" unpatched anyway, so
the paste transplants a naming the donor already had; and
naming outlives the yes/no margin (margin dead by 24-28,
naming alive at 28), so the naming path and the presence-
margin path are not the same readout. ChatGPT's caution on
exactly this point is borne out, now with controls.

## Scorecard against the registered predictions

- A (wholesale vs specific): neither clean answer; the
  dissociation above is the result. Recorded as a partial
  miss for both registered branches.
- B (the paste moves the originals' and choices' margins;
  reversed pitch left open): the paste moves all six
  wordings' margins, reversed pitch included.
- C (mediation): void by construction (above); prediction
  untested, not failed.
- D (if unpatched recipients say "beep", the 18/18 was the
  battery): recipients split 3 silent / 3 named-with-other-
  labels; donors 6/6 "beep"; the sham control separates the
  paste from the battery. The battery-artifact branch is
  ruled out; the rescue wording stays withdrawn in favor of
  the narrower transplant claim above.

## What changes in the ledger

- Property-specificity: replaced. The transplant is
  band-level for pitch (mid-window) and instance-level for
  onset, with the other property always partly transplanted.
- The naming claim: replaced by the controlled version in
  Part D.
- Onset's paste window: closes between layers 20 and 21
  (was "21-22").
- Open: the corrected mediation cell (restore post-block
  rows at every layer from the paste layer up).

## Correction note (Nov 4, 2026, after the council's fixpack audits)

Four statements above do not stand as written; the council's
audits (saved in council/council-replies-2026-11-04-series-
three-results.md, with Muse's verification) correct them:
(1) "The run's own printed readout matches the recomputation
exactly" is false: the log prints the pitch sham's own-read at
L19/L20 as -63%/-44%; the rows support this read's table, so
the log line is the wrong one, but the sentence above is
withdrawn. (2) The dissociation in Part A must be argued
within the sham_pitch rows (same paste: pitch readout flat,
onset readout full), not across arms; the shares sit on a
1/64-logit grid and the cross-arm comparison is underpowered
(p 0.054-0.087). Onset's sham moves overshoot their gap
(124-155%): "a move the size of the gap, in the donor's
direction." (3) Part D's framing is replaced: the sham crosses
onset while the real arm crosses pitch, so the contrast shows
"crossing pitch transplants naming, crossing onset does not";
a both-properties-matched donor control is still missing, and
only layer 28 compares real vs sham within one run. (4) The
regression cell Claude specified ran from the behavioral
file: within-band, the pitch lean does not track exact
frequency (corr 0.001 / 0.014), while the onset lean tracks
exact start time within band (-0.500 / -0.338). The lean is a
band meter for pitch and an instance meter for onset.
