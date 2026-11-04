# Index check + behavioral questions: build note (Nov 4, 2026)

Built on Kobby's tap ("cheap checks first"), following Claude's Nov 4 audit
order (1) and (2), with ChatGPT's readout point folded in.

Notebook: colab/project_k_qwen_indexcheck_colab.ipynb; builder:
colab/build_indexcheck_qwen_notebook.py. The runner's head (proof, load,
scorer, feed, token check, feed proof, baselines, captures,
patched_generate) is the Handover Correction rerun's own runner, taken
whole so the check tests the suspect machinery as built. Three surgical
changes only: a state_index override and an "all" mode in
patched_generate, a run-log tee (indexcheck_run_log.txt), and indexcheck_
output names. Runner compiles (48,262 chars); the notebook's embedded
runner was extracted and compile-checked after a regex replacement bug
(re.sub mangling the repr replacement) was caught and fixed with a lambda
replacement.

Part A cells (donors tone_00..03 into silence_00..03, PROMPT_BEEPS):
- Norm diagnostic: reproduces the forward's own last-position logits from
  hidden_states[36] raw and after THINKER.model.norm; the near-zero diff
  settles whether state 36 is pre- or post-norm in this stack.
- all_positions at layers 0, 18, 28, 35 (4 pairs each): after a full-row
  replacement the score must equal the donor baseline at every layer.
- final-only: layer 35 as built (state 36), layer 35 shifted (state 35),
  layer 34 as built, layer 28 as built (replication of the rerun's
  near-donor cell).
Output: indexcheck_results.csv with per-cell diff vs donor baseline.

Part B (same feed form as every Qwen battery; scored from first-token
logits with the Series Two yes scorer plus the normalized yes share over
Yes/No tokens):
- All 360 property tones (files + manifest from the property battery on
  Drive): PITCH_Q "Is the tone in this clip high-pitched? Answer yes or
  no." (truth: pitch_band high) and ONSET_Q "Does the tone start in the
  first second of this clip? Answer yes or no." (truth: onset_band early;
  bursts are 0.5 s at 0.5/2.5 s with +-0.1 s jitter, so early starts land
  at 0.4-0.6 s, inside the first second; late at 2.4-2.6 s).
  Output: behavior_property_results.csv (per ask: yes score, yes share,
  first token, three correctness flags).
- All 8 unpatched patching tones under PROMPT_OPEN (P3):
  openform_tones_unpatched.csv, the missing baseline beside
  patching_openform.csv.

Feed note, settled while building: the tracing/property/patching feeds
are one form (audio-only conversation, processor(text, audio)); the 250
audio positions in the token checks were the 10 s master files, the 4 s
battery clips give 100. No picture is in any of these feeds.
