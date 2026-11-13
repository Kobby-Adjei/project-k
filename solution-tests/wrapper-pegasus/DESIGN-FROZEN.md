# Kill test: wrapper fix (Path A) — design frozen before any call
Date frozen: Nov 11, 2026, before the run. Kobby's go: "Go" (kill test only).
Model: Pegasus 1.6, Twelve Labs Analyze, temperature 0, via custom.twelvelabs-temp4 (tests-only account).
Scope: Path A only. Path B (inside cut) is a separate Colab check and is NOT part of this run.

## Clips (existing files, picture carries no sound information)
- water: pilot2/swap/swap_water.mp4 (still picture + running water soundtrack)
- vacuum: pilot2/swap/swap_vacuum.mp4 (still picture + vacuum soundtrack)
- applause: docs-sounds-check/docs_applause_still.mp4 (still picture + applause)
- silence: pilot2/swap/swap_silence.mp4 (control, no sound)

## Arms (1 repeat per cell; temperature 0 repeats agreed almost every time in earlier runs)
- S, standard (1 call per clip): "Describe what happens in this video, including any sounds that are audible." (battery prompt P2, verbatim)
- R, ordinary re-ask control, budget-matched (3 calls per sound clip):
  R1 = P2 (same words as S, separate call)
  R2 = "What happens in this video? Mention any sounds you can hear."
  R3 = "Summarize this video and list any audible sounds."
  R counts as naming the sound if ANY of its 3 answers names it.
- W, wrapper (3 calls per sound clip, in order):
  W1 sound-only (battery prompt P3, verbatim): "What sounds can you hear in this clip? Describe only what is audible, without guessing from the pictures. If the audio is silent, say 'silent'."
  W2 yes/no own sound: "Is there the sound of {target} in this clip? Answer yes or no." Targets: water = "running water", vacuum = "a vacuum cleaner", applause = "applause".
  W3 carry-forward final: "A sound-only check of this same clip reported: '{W1 answer}' Now describe what happens in this video, including any sounds that are audible. Include the sounds from that sound-only report in your answer."
  W is scored on W3 only (the corrected final answer), with W1 recorded separately.
- Lures (1 call per sound clip): yes/no with the WRONG target (water asked "a vacuum cleaner", vacuum asked "applause", applause asked "running water").
- Silence control (4 calls): S (P2), W1 (P3), W3 (carry-forward built from its own W1), yes/no "running water".

Total calls: 3 sound clips x 10 + silence 4 = 34. Stop the run on HTTP 422 (allowance gone) and report partial.

## Bar, frozen before the run (kill test bar, NOT the public solution bar)
- W survives if W3 names the correct sound in at least 2 of 3 sound clips, AND W beats both S and R on those clips, AND W3 names no sound on the silence clip.
- A is killed as a fix if W3 names the correct sound in 0 or 1 of 3 clips, OR if R does as well as W (the gain is then just re-asking).
- Flag-only outcome: if W1 names a sound but W3 drops it again, the carry-forward failed; what remains is a flag, not a fix, and it will be reported as that.
- Scoring is from the saved raw answer files only, by whether the answer names the target sound (or a plain synonym: water / running water / tap; vacuum / vacuum cleaner / hoover; applause / clapping). No credit for vague words like "noise" or a wrong sound renamed.
