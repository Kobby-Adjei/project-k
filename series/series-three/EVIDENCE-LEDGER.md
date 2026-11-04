# Series Three: evidence ledger (freeze gate, Nov 4, 2026)

One page, three buckets. Every claim is scoped to the
tested setup: Qwen2.5-Omni-3B, synthetic tones and the
codec stimulus set, the pinned transformers 4.52.4 stack,
greedy first-token scoring unless stated. "In this model"
is part of every sentence below. The central distinction
of the series runs through the buckets: is the sound
ABSENT from the model's representations, or PRESENT but
not used correctly in the answer? For everything tested
here, the answer is the second.

## Established in the tested setup

1. The file is not the loss. Through one input door,
   the same sound compressed (the project's AAC recipe)
   and raw gives the same answers; the codec's only
   measurable effect is on naming the quietest sounds,
   one clip each. (Codec control, two rounds.)
2. The sound is present in the representations. Sound
   presence, pitch band, and onset stay linearly
   decodable at every layer, at the sound's positions
   and (from layer 1) at the final answer position,
   against 200-permutation nulls. Decodability is not
   use; the items below are the use tests.
3. The spoken failure is a misplaced cut, not a missing
   signal. Asked directly, the model answers at the
   base rate at the fixed 0.5 cut while its own margin
   separates the classes almost perfectly (pitch AUC
   1.0000, onset AUC 0.9988 on 360 tones). Across
   wordings, the spoken answer defaults to the
   affirmative or first-named option; the margin ranks
   the truth in every form but reversed pitch (0.6331).
4. The margin is causally set at the sound's positions
   in a depth window. Pasting a donor's audio-position
   states sets the recipient's margin to the donor's
   value, fully through about layers 18-20, halved by
   22, gone by 24, both properties, both directions,
   with gap-matched sham and position controls. The
   layer indexing behind every patch number is verified
   exactly (all-positions pastes reproduce donors at
   layers 0, 18, 28; the post-norm top cell is exact).
5. The state carries finer pitch than the answer uses.
   A graded probe read on patched states recovers the
   donor's exact frequency downstream of the paste at
   the audio positions (98.6% of low-pair rows within
   the probe's own error of the donor), while the
   yes/no margin does not follow jitter-scale pitch
   differences at all. Coarseness is in the readout of
   pitch, not in its representation.
6. In the fade zone, the effect routes necessarily
   through the later text positions. Holding the
   post-block positions to the recipient's own states
   at every layer from the paste up leaves exactly one
   readout step of the paste's effect at layers 19-22,
   both properties (restore-only control bit-exact:
   the hooks change nothing on an unpatched run).
7. Naming follows the donor across a pitch crossing,
   and only there. Under real pastes (donor high, same
   onset), six low recipients answer the donor's own
   word "beep" in 24 of 24 cells at layers 14-28; under
   donors matched on BOTH properties, 0 of 24 cells do,
   and recipients keep their own answer texture. This
   is a transplant (the donors name it unpatched
   anyway), not a rescue, and it outlives the margin,
   which is dead by layer 24.

## Supported but limited

- The scale split: the margin treats pitch at band
  scale and onset at instance scale. Strong form is
  the within-row version (one paste, the onset readout
  follows fully, the pitch readout flat); the
  cross-arm percentages rest on 6 pairs per arm and a
  coarse readout lattice, and onset's small moves
  overshoot their gap.
- Fine pitch reaches the final position late: the
  probe's following rate ramps with downstream distance
  (about 39% just after the paste to 87% late, low
  pairs). One instrument, and that instrument (the
  probe cell's) is weaker than the graded battery's.
- Presence decodability rests on the tracing round
  whose silence split leaked waveforms between train
  and test; the pitch and onset probes used clean
  instance splits and carry their claims alone.
- Speech: transcribed perfectly and denied as a sound
  (behavioral batteries), readable at every layer
  (round 2) — the reporting loss for speech is
  established behaviorally; the causal patch framework
  was run on tones, and its one speech arm used a
  single donor sentence.

## Unresolved, stated as limitations

- One model. Nothing here is claimed about any other
  model; the protocol exists so a second model can be
  run identically, and MiniCPM is the named leftover.
- Synthetic tones only inside the model. No household
  sound has been traced; the household study is
  behavioral so far.
- The post-block route is necessary, not shown to be
  the only one, and its carriers are unnamed (question
  words, template, and prefix positions are restored
  as a block).
- The top of the stack cannot be patched (the final
  tuple entry is the post-norm state); graded pitch
  error drifts mildly with depth, unexplained.
- Attention weights were dropped as evidence early
  (weight is not influence) and appear in no claim.
