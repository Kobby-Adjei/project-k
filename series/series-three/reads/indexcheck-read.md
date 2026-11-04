# The cheap checks, read (Nov 4, 2026)

Run: Nov 4, 2026, free lane (kadjei1), L4, exit 0, token check passed.
Files: indexcheck_results.csv, behavior_property_results.csv,
openform_tones_unpatched.csv, indexcheck_run_log.txt (all in the Drive
tracing-qwen3b folder; local copies beside this file). The notebook's
runner head is the Handover Correction rerun's own machinery, so these
cells test the suspect code as built. The NORM DIAGNOSTIC inside the
run failed on a dtype bug in my diagnostic code (float vs half); the
question it asked is settled below from the pinned library source
instead, which is stronger evidence.

## Part A: the index check. The break is at the last index only

All-positions patches (donor tone into recipient silence, 4 pairs):
the yes score equals the donor baseline EXACTLY (diff 0.0000) at
layers 0, 18, and 28, and fails only at layer 35 (diffs 0.33-0.39;
scores 0.5217 / 0.5211 / 0.4067 / 0.5457 against baselines 0.8811 /
0.8870 / 0.7986 / 0.8781). Final-position-only at 35 as built gives
the identical four scores as the all-positions paste at 35, which is
expected: after the last layer only the per-position norm and head
run before the first token, so other positions cannot matter there.

Final-position cells: at 35 with the donor index shifted to 35 the
scores move partway (0.53-0.60) but stay far from the donor; at 34 as
built they are near-exact (diffs 0.003-0.005); at 28 as built they
replicate the rerun's near-donor cell (diffs 0.018-0.042; the small
gap is expected, since layers 29-35 still process the pasted
position against the recipient's other positions).

## Why: state 36 is the post-norm final state, not a layer output

Read in the pinned transformers 4.52.4 source
(models/qwen2_5_omni/modeling_qwen2_5_omni.py, the thinker text
model's forward, lines 1948-1988): the loop appends the state BEFORE
each decoder layer runs, so tuple entry k (0-35) is the input to
layer k, i.e. the output of layer k-1 (entry 0 is the embedding
output). After the loop, the code applies the final norm
(hidden_states = self.norm(hidden_states)) and appends THAT as entry
36. The raw output of layer 35 is never in the tuple.

So the convention "state L+1 = output of decoder layer L" holds for
L = 0..34 and breaks exactly at the top: states[36] is
norm(output of layer 35). Every "layer 35" patch in this project
pasted an already-normed vector into the output slot of layer 35,
and the model then normed it a second time. A double-normed state is
a distorted state, which is what the ~0.52 scores were measuring.

What this changes:
- Layers 0-34 in every surgery sweep were patched correctly; their
  rows stand as labeled. Only the layer-35 rows are artifacts: the
  original sweep's layer-35 rows, and the rerun's final@35 cells
  (mean 0.4988, "partial rescue at 35") are WITHDRAWN as double-norm
  artifacts. The corrected relay wording (audio slots 0-18,
  post-block 14-24, final from 23) does not rest on layer 35; its
  final-position evidence is layers 23, 24, and 28, all valid.
- The probes (tracing round 2, step 5) read states 0-36. States
  0-35 are labeled correctly under the fixed convention. State 36
  was labeled "output of layer 35"; it is the post-norm final state.
  A probe reading it is still a valid readout of a real state in the
  pipeline, so the decodability results stand; only the label was
  wrong, and no claim turned on that one point (accuracy was 1.00
  at every state).
- Claude's sanity argument (Nov 4 audit) is confirmed and now has
  its mechanism. The index check was the right call: one battery
  localized the break to a single index and explained it.

## Part B: asked directly, the model answers yes to everything

The 360 property tones (the same files the step 5 probes read at
1.000 accuracy) were asked two yes/no questions through the same
feed, scored from the first-token logits. 720 asks, all recorded in
behavior_property_results.csv.

Onset ("Does the tone start in the first second of this clip?"):
the model's first token is Yes on 360 of 360 asks. Accuracy by the
spoken word and by the normalized yes share is exactly the base
rate: 1.000 on early tones, 0.000 on late tones, 0.500 overall.
Pitch ("Is the tone in this clip high-pitched?"): first token Yes
on 343 of 360 asks; by yes share, 1.000 on high tones and 0.089 on
low tones (0.544 overall).

The logit balance does lean the right way. Mean yes share: onset
0.720 when the tone truly starts early vs 0.607 when late; pitch
0.671 when truly high vs 0.560 when low. So a graded trace of the
property reaches the answer position, but it never crosses the
decision: the spoken answer is a near-constant yes regardless of
which property value the clip carries.

CORRECTION (Nov 4, 2026, Claude's follow-up audit, verified by
recomputation from the CSV): the paragraph above undersells what
the file shows, and the failure-of-use reading of it is wrong.
The yes/no margin does not merely lean: it separates the classes
almost perfectly. Pitch yes_share ranges do not overlap at all
(low 0.3886-0.5927, high 0.6114-0.7719), AUC 1.0000, and a
threshold at 0.602 classifies all 360 asks correctly. Onset
separates at AUC 0.9988 (threshold 0.6843 gives 357 of 360). The
fixed 0.5 threshold sits below the entire yes_share distribution;
that misplaced cut, not a failure to use the information, is what
produces the wrong answers. The defensible claim: on these tones
the yes/no decision is yes-biased; pitch and onset separate the
model's own margin perfectly, so the threshold, not the
representation or its transmission, is at fault. Scope: this
model, these tones, one wording per question; whether the bias
is a property of the question form is untested (the
wording-variation battery is the check). "Failure of use" for
pitch and onset is withdrawn. (One data note from the audit:
the emitted word tracks the yes_share, not the raw yes_score;
the two disagree on 50 of 720 rows, and one exact tie at 0.5000,
prop_tone_0_85's pitch ask, is the whole difference between the
0.089 and 0.0944 low-pitch figures.)

## Open form: the naming-gap control is closed

All 8 unpatched patching tones under the open question (P3):
tone_05 and tone_07 answer "beep"; the other six answer "silent".
Exactly the split in patching_openform.csv (the patched file). The
baseline naming rate is 2 of 8 with no patch anywhere, so the
patched runs' naming pattern was not a paste artifact.

## What stands now

- The patch convention is verified exact at three widely spaced
  layers (0, 18, 28); layers 0-34 stand as labeled, and layer 35
  rows in earlier files are annotated as artifacts. One cell
  remains open by the audit's design: pasting donor entry 36
  into the post-norm slot at the final position, which must
  return the donor baselines exactly and would close the account
  empirically.
- The behavioral result, as corrected: the yes/no answers are
  threshold-wrong, not information-blind. Pitch and onset
  separate the model's own margin at AUC 1.00 and 0.999; the
  yes/no cut sits below the whole distribution.
- Next in the council's order: the one-cell top check, a
  wording-variation battery (ChatGPT's addition), then the
  property patch, then MiniCPM.
