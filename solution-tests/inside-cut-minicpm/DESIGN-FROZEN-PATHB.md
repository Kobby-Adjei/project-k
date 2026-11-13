# Kill test: Path B (inside cut) transfer to real sounds — design frozen before any run
Date frozen: Nov 11, 2026, before the run. Kobby's go: "Next: test Path B, the inside cut."
Model: MiniCPM-o 2.6, pinned revision 06849bfd36da94da1f5a88fa17e8ba63f08a4c4d, bf16, paid kobbytti Colab lane, L4.
Council's test, as given Nov 11, 2026: apply the cut from the tone data to real sounds and no-sound clips WITHOUT recalibrating it; if the score is below about 0.8, Path B is dead.

## What "the cut" is, exactly
In the MiniCPM behavioral run (Series Three cell 3), every yes/no ask was scored from the model's own next-token numbers: the yes share = P(Yes tokens) / (P(Yes) + P(No)) at the answer position, token sets Yes [7414, 9454], No [2308, 2753], summed over surface forms, through the manual forward. The spoken answer follows the share at the default cut 0.5.
Tone results (360 tones per form, from behavioral_minicpm_results.csv via minicpm-behavioral-read.md):
- Original pitch form: AUC 0.9476, best cut 0.7773 (0.9000 right at that cut).
- Original onset form: AUC 0.9831, best cut 0.4378 (0.9472 right at that cut).
The cuts are form-specific; there is no single tone cut. Both are carried into this test, fixed, alongside the default 0.5. NOTHING is fitted on the real sounds.

## Transfer set: 18 asks, audio-alone feed (the behavioral runner's own feed shape)
Stimuli: the first 4.000 s of each clip's audio track, 16 kHz mono WAV, loudness untouched (the swap and Series Two files were RMS-matched when built). Feed: per-second audio chunks + the question, no picture, the Series Two audio-alone shape the behavioral runner uses. The question is the Series Two Test 3 form, verbatim: "Is there the sound of {target} in this clip? Answer yes or no."
Present (truth Yes), 9 asks:
1. swap_water.mp4, target "running water"
2. swap_vacuum.mp4, target "a vacuum cleaner"
3. swap_beeps.mp4, target "beeping"
4. swap_sizzle.mp4, target "sizzling"
5. swap_speech.mp4, target "a person speaking"
6. swap_mixed.mp4, target "a vacuum cleaner"
7. series2_gray_water.mp4, target "running water"
8. series2_gray_vacuum.mp4, target "a vacuum cleaner"
9. series2_gray_beeps.mp4, target "beeping"
Absent (truth No), 9 asks:
10. swap_silence.mp4, target "running water"
11. swap_silence.mp4, target "a vacuum cleaner"
12. swap_silence.mp4, target "beeping"
13. swap_silence.mp4, target "sizzling"
14. swap_silence.mp4, target "a person speaking"
15. swap_water.mp4, target "a vacuum cleaner" (lure)
16. swap_vacuum.mp4, target "beeping" (lure)
17. swap_beeps.mp4, target "running water" (lure)
18. swap_sizzle.mp4, target "a person speaking" (lure)

## Bar, frozen before the run
- Ranking transfer: AUC of the yes share, present vs absent. Below 0.80, Path B is dead (council's number).
- Cut transfer: a tone cut (0.7773 or 0.4378) counts as a working fix only if its accuracy on these 18 asks BEATS the default 0.5 accuracy on the same asks. Matching or losing to the default means the cut does not transfer as a fix, whatever the AUC.
- Both numbers are reported whichever way they land. No other threshold will be tried after seeing the scores; a best cut computed on these 18 asks may be printed for the record, clearly labeled as fitted on the test itself and NOT evidence of a fix.
- Scope: one model, 18 asks, audio-alone, 4-second cuts. This tests whether the tone-derived cut story transfers to real-sound presence. It does not test open naming, and it does not test Qwen (Qwen's tone cuts, 0.602 pitch / 0.6843 onset, are a separate transfer check if this one lives).
