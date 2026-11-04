# Patching (the surgery): build note (Qwen 3B)

Built Nov 3, 2026. Builder: `colab/build_patching_qwen_notebook.py`. Notebook:
`colab/project_k_qwen_patching_colab.ipynb` (11 cells, 72,226 bytes). Built and
verified locally only: not uploaded, not run. The run launches separately; the
clips are uploaded to Drive `tracing-qwen3b/patching_clips/` by the parent after
this build, and the notebook's Step 2 resolves them from there.

## Design as built

Structure mirrors the tracing round 2 build cell for cell: intro, Step 1 (the
Series One notebook's own setup cells, reused, plus the stack proof:
transformers exactly 4.52.4, exit 3), Step 2 (fetch the clips), Step 3 (write
the runner, run check mode), Step 4 (battery mode), closing (what to send
back). The runner is a plain script under the session's own Python with
check/battery modes, the tracing runners' load candidates (3B fp16 first, 7B
AWQ second, talker off), full candidate tracebacks to `patching_load_log.txt`
in the tracing folder, fixed seed 20261105, outputs saved to the Drive folder
tracing-qwen3b and to /content.

Check mode, in order: stack proof; load; **decoder layer resolution** (every
module list in the loaded model is printed; the decoder is the list of exactly
36 modules under the thinker, path and count printed, hard stop exit 12 on any
other outcome, resolved at runtime and never assumed); the **token check**
(tracing round 2's histogram design on tone_00 and speech_00, audio id derived
from the histogram as the highest-count candidate, both clips must agree, and
the count must be exactly **250**, hard stop exit 11, transcript saved as
`patching_token_check.txt` either way; one thinker forward then cross-checks
the hidden states' layer count against the resolved list, exit 12 on
disagreement); the **feed proof** (all 20 clips against `manifest.json`: file
SHA-256 prefix, decoded seconds, sample rate, RMS per clip, hard stop exit 7).
The manifest's `sha256_16` values are prefixes of the files' own SHA-256 of
their bytes: verified locally for all 20 files before this build.

Battery mode: **baselines** for all 20 clips (presence score and answer word
under each clip's own question; the open P3 answer for tone_00, silence_00,
speech_00 as extra rows), saved to `patching_baselines.csv`. **Captures**: 16
donor forwards (8 tone and 4 silence under the beeping question, 4 speech under
the speaking question), full hidden states held on CPU as float16 with each
donor's row and audio positions. **The patch**: a forward hook on decoder layer
L replaces that layer's output at the chosen positions with the donor's layer-L
output (hidden_states index L + 1; index 0 is the embedding output), donor
position by donor position in order. The hook patches only the full-length
forward (the prefill that decides the first generated token) and leaves the
one-token decode steps alone, guarded by the sequence length; it is always
removed after the run. Donor and recipient always run under the same question,
the direction's scored question, so a patch moves sound information, never
question information; a pair whose rows differ in length stops the run
(exit 12) rather than patching across a misalignment.

**Sweep** (each run appends its row to `patching_results.csv` in both homes
the moment it finishes; a fresh battery first renames any earlier results file
aside with a timestamp): tone_to_silence 8 pairs x 36 layers = 288;
speech_to_silence 4 x 36 = 144; silence_to_tone 4 x 36 = 144 (reverse control);
tone_to_tone 4 x 36 = 144 (sham: donor tone_i into recipient tone_{i+1 mod 8});
wrong_positions 4 pairs x layers {0, 7, 14, 21, 28, 35} = 24 (patch at the
non-audio positions only). Total **744 rows**. Then the **best-layer open-form
check**: per-layer mean tone_to_silence lift over each recipient's own baseline
is printed for all 36 layers, and at the best layer the 8 patched silence runs
answer the open question into `patching_openform.csv`, with the unpatched
silence_00 open answer printed beside them. Closing prints: "patching done",
the plan's two patching predictions quoted exactly, and the output filenames.
No verdict on the predictions is printed anywhere.

## The questions, copied not retyped (with sources)

- "Is there the sound of beeping in this clip? Answer yes or no." Built by the
  Series Two constructor, `build_series2_qwen_notebook.py` lines 1304-1305
  (`prompt = (f"Is there the sound of {target} in this clip? " "Answer yes or
  no.")`), with target "beeping" from its TARGETS table, line 1280.
- "Is there the sound of a person speaking in this clip? Answer yes or no."
  Same constructor, target "a person speaking", TARGETS line 1279.
- Open question (P3): "What sounds can you hear in this clip? Describe only
  what is audible, without guessing from the pictures. If the audio is silent,
  say 'silent'." Extracted at build time from `pilot2/swap/run_swap.py` lines
  51-54 (the 'P3' entry of PROMPTS) with the codec builder's own extraction
  method and asserted against the expected literal. Note: P3 does not appear in
  the Series Two Qwen runner itself (its open form is the JSON prompt); P3 is
  the swap study's question, carried by the codec and equal runners, and it is
  the open question the brief names.
- Scorer: `yes_probability`, copied from `build_series2_qwen_notebook.py`
  lines 995-1010 (softmax over the vocabulary of `scores[0][0]`, summed over
  the yes ids), with the yes ids read from the model's own tokenizer exactly as
  there (first token of "Yes" and of " Yes", lines 830-837). The generate call
  is that runner's ask path (lines 1171-1178): greedy, `do_sample=False`,
  `return_dict_in_generate=True`, `output_scores=True`, cap passed as
  `thinker_max_new_tokens`, only the newly written tokens decoded.
- Token caps: presence asks 16 (Series Two Test 3's yes/no cap, line 1306);
  open asks 300 (the codec Qwen runner's open P3 cap,
  `build_codec_qwen_notebook.py` line 105).
- In the results CSV, `answer_word` holds the full decoded answer of that
  generate call, stripped, the Series Two runner's raw-answer convention; its
  first token is the token the score reads.

## Attention implementation choice

No `attn_implementation` is passed at load, so the model loads with the
library default (sdpa). Patching reads no attention weights anywhere; the
hooks act on decoder layer outputs, not on attention. This is also the door
the Series Two asks themselves ran through (their load call passes no
attention implementation either), so the baselines are read under the same
conditions as the Series Two rows they extend. Tracing round 2 loaded eager
for one stated reason only: sdpa returns no attention weights and that build
had to publish an attention readout. This build has no such output, so the
default stands, and the runner's proof lines print the reasoning in the open.

## Deviations from the brief, and flags

1. **The 250 assert is built exactly as specified; one tension is flagged,
   not silently resolved.** The brief fixes the expected audio-position count
   at 250 for these 4.0 s clips (Steps: "Patching replaces ALL audio
   positions (250 for these 4 s clips)" and the Step 1 assert). The token
   check therefore requires exactly 250 in both the tone_00 and speech_00
   rows and hard-stops (exit 11) otherwise, with the full histogram printed
   and saved as the evidence. The flag: tracing round 2's token check
   measured 250 audio positions on the codec masters, which are 10.0 s
   clips, roughly 25 audio tokens per second; the same rate over 4.0 s
   would predict roughly 100 positions. If the histogram shows that, the
   check fails loudly on the first run and the constant
   (`EXPECTED_AUDIO_POSITIONS`, one literal in the builder's spec block and
   injected into the runner) is the single place to correct. Nothing
   downstream can be quietly wrong either way: captures and recipients also
   assert the same count before any patch.
2. Step 1's addition proves the transformers stack only; the scikit-learn
   proof from the tracing build is dropped because this build trains no
   classifiers. The addition's text says so.
3. The load log and the token-check transcript are named for this build
   (`patching_load_log.txt`, `patching_token_check.txt`) in the same tracing
   folder and the same format as round 2's, rather than reusing round 2's
   filenames, so the two runs' records cannot overwrite each other.
4. The speech clips are four copies of one source window (the manifest shows
   one `source_window` and one hash for all four), so speech_to_silence is
   effectively one donor sentence in four identical copies; the sweep keeps
   the brief's 4-pair shape, and any read of that direction should treat it
   as one donor, four recipients. Noted here because the CSV will look like
   four donors.

## Verification results (local, before any upload)

- Builder runs clean and writes the notebook: 11 cells, nbformat 4,
  `nbformat.validate` passes, 72,226 bytes.
- Every code cell in the notebook parses; the embedded runner source
  (43,565 chars) parses.
- Em dashes in the built notebook: **0** (en dashes also 0; the builder file
  itself is also clean).
- The three prompt strings appear verbatim in the built runner (checked by
  string search on the extracted runner source), the token-check assert value
  is 250, the expected results count 744 is printed by the closing sweep
  lines, the "patching done" print and both quoted predictions are present,
  the layer-resolution print, the per-run CSV append, and the prefill-only
  hook guard are all in the code, not just intended.
- Sweep row math, from the builder's spec block: 288 + 144 + 144 + 144 + 24
  = 744, asserted in the builder at build time.

## What to watch in the run

Check mode must show the derived audio id at exactly 250 in both histograms
(see the flag above), the resolved layer path with 36 layers, and 20 clean
feed-proof lines. Battery mode: 23 baseline rows, 16 captures, 744 appended
result rows with progress printed per layer, the per-layer lift table, and
the 8 open-form answers at the best layer.

## Parent spot check (Nov 3, 2026, after the build)

The flag in deviation 1 is settled by the tracing round 2 builder itself
(build_tracing2_qwen_notebook.py, lines 33, 216, 854, 1515): round 2's clips
were exactly 4.0 s each, and its token check measured 250 audio positions on
those 4.0 s clips. The "10.0 s codec masters" premise in the flag was a
conflation with the codec battery; at 4.0 s through this feed path the model
emits about 62 audio tokens per second, not 25. EXPECTED_AUDIO_POSITIONS = 250
stands as built. One real difference remains: tracing fed the still picture
alongside the audio (row total 287); this build feeds audio only, so row
totals will be shorter while the audio count should still read 250. The check
asserts the audio count, not the total, so this is consistent.
Notebook verified: 11 cells, em dash count 0 (checked on the file bytes).

## Flag resolution (parent, Nov 3, 2026)
The 250-position assert stands. The doubt came from assuming the 250 count belonged
to the 10.0 s codec masters; it does not. Tracing round 2's own token-check
transcript (tracing2_token_check.txt, saved) records count 250 for id 151646
'<|AUDIO|>' on tracing2's clips, and its builder fixes those clips at exactly
4.0 s (CLIP_SAMPLES = SR * 4). These patching clips are also 4.0 s at the same
sample rate and feed path, so 250 is the evidence-based expectation. Audio-only
rows will total shorter than tracing2's 286-287 (no picture tokens).

## Correction (Nov 3, 2026, after the first run's token check)
The flag was right and the parent's resolution was wrong. The first patching run's
token check measured 100 audio positions per 4.0 s clip through the audio-only feed
(tone row 137 total, speech 138), and stopped the run cleanly (exit 11) before any
patching. Tracing round 2's 250 count came through its own feed path (rows of
286-287); the non-audio part is the same 37 positions in both, so the difference is
purely the audio token rate of the door used. Each test's check verifies its own
feed, so round 2's readout is unaffected; position counts are not comparable
across the two tests. Fix: EXPECTED_AUDIO_POSITIONS 250 -> 100 in the builder,
notebook rebuilt (11 cells, no em dashes), Drive copy replaced, run relaunched.

## Open-form stage fix (Nov 3, 2026, after the first full run)
The sweep completed (744 rows) but the battery exited 12 in the best-layer
open-form stage: that stage reused the donor capture taken under the short
presence question while the recipient ran under the long open question, and
patched_generate's one-to-one row guard correctly refused the mismatched rows.
Fix: the stage now captures each tone donor under the open question first and
patches from that capture. Notebook rebuilt and Drive copy replaced; the whole
battery reruns (the earlier results file is renamed aside by the notebook; the
parent holds the first sweep's CSV locally).
