# Handover Correction step 2: build note (Nov 3, 2026)

The rerun of the wrong-positions cell as three independent constructions.
Builder: colab/build_handover_rerun_notebook.py, derived from the patching
builder with the sweep replaced. Notebook:
colab/project_k_qwen_handover_rerun_colab.ipynb (11 cells, 0 em dashes),
in the Drive tracing-qwen3b folder.

Design: tone donor into silence recipient, 4 pairs, presence beeps question,
yes score and emitted answer word recorded for every run. Three patch modes,
each a strict subset of the non-audio positions, so none can assemble the
complementary hybrid state (donor audio + recipient non-audio) that made the
original cell identical to the reverse control:
- final: only the last row position (the one that predicts the first answer
  token).
- pre: only the positions before the audio block.
- post: only the positions after the audio block, final position excluded.
Layers: the original six (0, 7, 14, 21, 28, 35) plus the boundary band
15 to 24. 4 pairs x 15 layers x 3 modes = 180 runs. Audio contiguity is a
mechanics check inside the new modes (exit 12 if the audio block is not one
contiguous span). Token check (100 audio positions), feed proof on all 20
clips, baselines, and captures are the surgery notebook's machinery unchanged.
Outputs: handover_rerun_results.csv, handover_rerun_baselines.csv,
handover_load_log.txt, handover_token_check.txt, all in tracing-qwen3b.

Registered expectation (printed by the notebook itself): if the tone's
deciding content truly sits in the non-audio positions at the late layers,
one subset alone (most plausibly final) should rescue at layers 21 to 28. If
no subset rescues anywhere, the original rescue was the hybrid identity and
nothing more.
