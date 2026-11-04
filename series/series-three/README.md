# Series Three: the inside tests

The question: is the sound lost in the file, or inside
the model, and if inside, exactly where?

Series One and Two test models from outside. This series
opens one of them, Qwen2.5-Omni-3B, and traces the sound
through its layers: probes that read the model's inner
states, and surgery that pastes one clip's inner states
into another clip's run and watches the answer move.

Everything here is scoped to that one model, the pinned
transformers 4.52.4 stack, and synthetic tones built for
exact ground truth, plus a codec control set. No claim in
this folder is about any other model.

## The short version of what was found

The sound is in there. Sound presence, pitch, and onset
stay readable in the inner states at every layer. The
answers fail anyway: the model's yes/no word is a fixed
cut on an inner lean that ranks the truth almost
perfectly, and the cut is misplaced. Pasting a donor's
states at the sound's positions moves the recipient's
lean to the donor's value in a window of layers (full
through about 18-20, gone by 24). The pasted state
carries the donor's exact pitch, finer than the lean can
read. In the fade zone the effect reaches the answer
only through the text positions after the sound.
Crossing pitch in a paste also transplants the donor's
naming of the sound; a paste matched on both pitch and
timing transplants nothing. EVIDENCE-LEDGER.md states
each claim at its strength, in three buckets: proven,
supported but limited, unresolved.

## What is in this folder

- EVIDENCE-LEDGER.md: the one-page claims ledger, the
  frozen record of what this series claims.
- PROTOCOL.md: the battery as a model-agnostic recipe,
  corrected versions only, so a second model can be run
  identically. The void designs are listed there so
  they are never rebuilt by accident.
- SERIES-THREE-JOURNEY.md: the full record, timeline,
  corrections, and the claims ledger as it moved.
- reads/: one read per battery, written from its result
  files (tracing rounds, surgery, the handover
  correction, property probes, index check, graded
  probes, wording battery, property patch, fixpack,
  probe cell, corrected mediation, naming control).
- results/: the result files themselves (CSVs,
  manifests, token checks, run logs). Round 1's probe
  files are included and are void; its read says why.
- codec/: the codec-loss control (same sound raw and
  compressed, through one door), its summary, and the
  stimulus provenance with SHA-256 hashes.
- build-notes/: how each notebook was built and what
  broke on the way.
- The notebooks and their builders are in ../../notebooks/
  with the other series' runners.

Not here: the extracted inner-state features for the
tracing rounds (a 156 MB array), and the stimulus media.
Both live in the project Drive; the manifests and hashes
in results/ and codec/provenance.md are enough to rebuild
the stimuli exactly.

## Status

Frozen Nov 4, 2026 (see FROZEN.md). The same inside
tests on MiniCPM-o 2.6 are the written leftover: they
will run under PROTOCOL.md, starting with that model's
own index and span checks, when they are taken up.
