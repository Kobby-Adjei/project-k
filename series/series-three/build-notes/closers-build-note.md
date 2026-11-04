# The two closers: build note (Nov 4, 2026)

Council follow-up audit items, Kobby's go ("run both closers, then
build the property patch"). Notebook:
colab/project_k_qwen_closers_colab.ipynb; builder:
colab/build_closers_qwen_notebook.py. Runner head is the index-check
runner's head whole (proof, load, scorer, feed, token check, feed
proof, baselines, captures), names switched indexcheck_ -> closers_;
new Parts 1 and 2 appended. Runner compiles (48,218 chars); the
notebook's embedded copy was extracted and compile-checked; one gap
caught pre-flight (NO_TOKEN_IDS lived in the cut Part B; it is now
defined in Part 2).

Part 1, the one-cell top check (Claude): forward only. For each of
tone_00..03 and silence_00..03: clone the recipient's hidden-state
tuple from thinker_forward, replace entry 36 at the final position
with the donor's, read the yes score from THINKER.lm_head. Cells:
4 self (construction sanity), 4 tone into silence, 4 silence into
tone. The pasted score must equal the pasted clip's generate
baseline exactly (donor baselines 0.8811 / 0.8870 / 0.7986 / 0.8781;
silence baselines about 0.3257). Output: topcheck_results.csv.

Part 2, the wording battery (ChatGPT): the 360 step-5 property
tones, same feed, first-token-logit scoring. Four forms:
pitch_reversed (low-pitched? yes/no; truth low), onset_reversed
(starts after the first second? yes/no; truth late; the
first-second line is clean for these bursts), pitch_choice
(high or low; scored on the high share of the high/low token
pair), onset_choice (early or late; early share). Token id sets
for the choice words are built from the tokenizer in-run (bare,
spaced, capitalized forms). Reporting per the corrected standard:
accuracy by word, by normalized share at 0.5, and AUC of the
share. Output: wording_results.csv (1440 rows). Registered
prediction: if the bias lives in the yes/no form, the reversed
questions mirror it and the forced choices score high; if the
forced choices also sit at chance, the threshold account needs
revision.
