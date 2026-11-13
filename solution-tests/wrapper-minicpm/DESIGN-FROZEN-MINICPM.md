# Kill test: wrapper fix (Path A) on MiniCPM-o 2.6 — design frozen before any call
Date frozen: Nov 11, 2026, before the run. Kobby's go: "Run the same test on MiniCPM" (kill test only).
Model: MiniCPM-o 2.6, pinned revision 06849bfd36da94da1f5a88fa17e8ba63f08a4c4d, bf16 full form, greedy (sampling=False), omni chat feed from the sealed MiniCPM recipe (1 frame per second + matching 1s of 16 kHz audio per unit, frames inside 448x448, max 15 units). Lane: paid kobbytti Colab, L4 (8B full precision per the routing rule).

## Same test as the Pegasus run, same bar
Identical clips, prompts, arms, and scoring as kill-test-wrapper/DESIGN-FROZEN.md:
- Clips: swap_water.mp4, swap_vacuum.mp4, docs_applause_still.mp4, swap_silence.mp4 (control).
- Arms per sound clip: S (P2), R1 to R3 (ordinary re-asks, budget-matched), W1 (P3 sound-only, verbatim), W2 (yes/no own sound), LURE (yes/no wrong sound); then W3 (carry-forward final quoting that clip's own W1 answer). Silence: S, W1, W2, W3.
- 28 asks total, 1 repeat per cell (greedy decoding; temperature 0 repeats agreed almost every time in earlier MiniCPM runs, and MiniCPM's Series Two repeats were identical on every cell).
- Bar, frozen: wrapper survives if W3 names the correct sound in at least 2 of 3 clips, beats both S and R, and names no sound on silence. Killed as a fix if W3 hits 0 or 1 of 3, or R matches W. Flag-only outcome reported as flag-only.
- Scoring from the saved results CSV only, same synonym rules as the Pegasus scoring script.
- Feed difference, stated plainly: Pegasus received the whole file through the API; MiniCPM receives the sealed unit feed above. That difference is part of the record, as in every earlier MiniCPM run.
