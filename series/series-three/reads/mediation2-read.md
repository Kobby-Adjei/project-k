# The corrected mediation test: read (Nov 4, 2026)

Run Nov 4, 2026, free lane (kadjei1), L4, exit 0, token
check passed. File: `mediation2_results.csv` (384 rows =
12 pairs x 2 questions x 4 layers x 4 arms). Design:
`mediation2-build-note.md`. All numbers recomputed from
the CSV by Muse and match the run's printed readout
exactly.

## The hook validation the void cell never had

The restore_only arm (the recipient's post-block states
restored into its own unpatched run at every layer from
19 up) returns the baseline EXACTLY: max |share - baseline|
= 0.0000 across all 96 rows. The restoration machinery
changes nothing when there is nothing to block. The void
cell's failure was the design (restoring at one layer,
where the rows are pre-paste by construction), not the
hooks; that is now proven, not assumed.

## The result

Mean toward-donor move of the lean, 6 pairs per cell:

| direction | layer | audio | combined | sham_combined |
|---|---|---|---|---|
| pitch (gap 0.098) | 19 | +0.057 | +0.004 | +0.001 |
| | 20 | +0.052 | +0.003 | +0.001 |
| | 21 | +0.031 | +0.004 | +0.000 |
| | 22 | +0.029 | +0.004 | +0.002 |
| onset (gap 0.127) | 19 | +0.119 | +0.004 | -0.001 |
| | 20 | +0.121 | +0.004 | -0.001 |
| | 21 | +0.018 | +0.005 | -0.001 |
| | 22 | +0.019 | +0.004 | -0.001 |

Holding the post-block positions (the text positions after
the audio block, final position excluded) to the
recipient's own states at every layer from the paste layer
up removes 93-97% of the paste's effect, at every layer in
the fade zone, for both properties. The sham paste with
the same restoration moves nothing, and the combined arm's
cross-question reads are also zero (+0.0015 / -0.0042).

The audio arm, measured in this same run, reproduces the
fade curve from one arm in one load: pitch at 58 / 53 /
32 / 30 percent of its gap at layers 19-22; onset at 94 /
95 / 14 / 15 percent — the onset window closing at 21, as
the fixpack found.

## What it licenses, and what it does not

Licensed: in the fade zone, the paste's effect on the
answer routes necessarily through the post-block positions.
If those positions are held to what the recipient would
have had anyway, the donor's content at the audio positions
cannot reach the answer. This is the relay, tested the way
the council specified, and it is consistent with the
handover rerun's post-block window (positions after the
block flipping answers at layers 14-24).

Not licensed, per the caveat Claude attached to the design
itself: the restoration is a sweeping intervention. It
blocks anything the pasted audio states would write into
the post-block stream at every later layer, and it re-runs
each layer's attention over recipient content. So the claim
is "a necessary route," not "the only route," and not a
single hand-off point. The windows overlap; the relay is a
region, not a layer.

## Correction note (Nov 4, 2026, the council's third audit)

"Removes 93-97% of the paste's effect, at every layer" is
wrong at layers 21-22 and is replaced. Per-pair suppression
recomputes to: pitch 93.2 / 93.8 / 84.9 / 84.7 percent and
onset 97.1 / 96.6 / 74.1 / 79.1 percent at layers 19-22.
The right statement is in readout steps: the combined arm's
residual is exactly ONE step of the yes-share's 1/64-logit
lattice at every layer (the file holds 51 distinct share
values; combined deltas quantize at 0 and about +-0.004),
which is 3-7% of the paste effect at layers 19-20 and
15-25% at layers 21-22, where the effect itself is fading.
The residual is not a second route: the same one-step
offset appears in the cross-question cells with the
opposite sign, so it is a per-pair-set constant, not
property-specific leftover donor content. The mechanism
claim (a necessary route through the post-block positions)
stands, and is cleaner without the percentage.
