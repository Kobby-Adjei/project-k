# Series Three, MiniCPM chapter (addendum)

The Qwen2.5-Omni-3B chapter of Series Three is frozen (see
../FROZEN.md) and is not edited by this addendum. In
November 2026 the same inside tests were run on
MiniCPM-o 2.6 under the frozen protocol, all seven cells,
and every number was recomputed from the delivered result
files.

The comparison lives in minicpm-beside-qwen.md. The short
version: the anatomy is the same in both models. The paste
window closes at about the same relative depth (about 54
percent of the stack in MiniCPM, about 53 percent in
Qwen). The naming transplant works in both (9 of 9 flips
in MiniCPM, matched controls 0 of 12). Exact pitch stays
readable at depths where pastes no longer move the answer:
present, readable, unused.

Scope note, applying to both chapters: the tone results
were measured on synthetic tones under the tested scoring.
Tone AUC shows recoverable low-level acoustic features,
not semantic identification. On real sounds, the tone cut
did not transfer (see ../../solution-tests/inside-cut-minicpm):
yes share AUC 0.7222, best accuracy 13 of 18, with present
sounds scoring below silence. That failure belongs beside
the tones claim wherever the claim is stated.
