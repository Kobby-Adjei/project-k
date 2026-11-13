# Docs sounds check on Qwen2.5-Omni-3B

- Date: Nov 6, 2026
- Model and settings: Qwen2.5-Omni-3B, fp16, greedy decoding
- Ran where: Google Colab, free lane (kadjei1), T4 GPU (L4 was not offered in that session)
- Files used: docs_birds_still.mp4, docs_birds_gray.mp4, docs_applause_still.mp4, docs_applause_gray.mp4, docs_fireworks_still.mp4, docs_fireworks_gray.mp4, swap_silence.mp4 (the muted control)
- Questions asked (verbatim): hearing form: "What sounds can you hear in this clip? Describe only what is audible, without guessing from the pictures. If the audio is silent, say 'silent'." Yes or no form: "Is there the sound of {target} in this clip? Answer yes or no." with targets "birds chirping", "applause", "fireworks"
- What came back: Hearing form, 3 repeats each, still picture: birds: "bird vocalization, wind, traffic noise" 3 of 3. Applause: "applause, crowd noise, music" 3 of 3. Fireworks: "wind, thunder, rain" 3 of 3 (a sound reported, misnamed). No clip was called silent. Yes or no: sound present Yes 12 of 12 (yes scores: birds 0.9212 still, 0.9340 gray; applause 0.9653, 0.9863; fireworks 0.5540, 0.5871). Silent file No 6 of 6. Wrong-sound lures No 6 of 6. 33 asks, clean run, every feed proved in print first.
- What it showed: On the same files and questions where Pegasus 1.6 reported nothing, this model named two of the three sounds and answered yes to all three. The flat denial is Pegasus's, not shared. Predictions written before the run all held.
- Source files: ~/workspace/project-k-pilot/docs-sounds-check/qwen3b/docs_qwen3b_results.csv; read at ~/workspace/project-k-pilot/docs-sounds-check/qwen3b/qwen3b-read.md; plan at ~/workspace/project-k-pilot/docs-sounds-check/qwen3b-plan.md
