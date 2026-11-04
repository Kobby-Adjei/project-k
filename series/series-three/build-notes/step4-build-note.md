# Handover Correction step 4: build note (Nov 3, 2026)

Confound verified in the original code before rebuilding (build_codec_qwen
notebook, decode_for): Arm A built a video-element conversation and called
the processor with use_audio_in_video=True; Arm B built image + audio
elements with use_audio_in_video=False. The codec contrast mixed the codec
with the input path, as the council alleged.

The fix (builder colab/build_codec2_qwen_notebook.py; notebook
project_k_qwen_codec2_colab.ipynb in the Drive codec-control folder):
three arms, 7 sounds x 3 arms x 2 forms x 2 repeats = 84 asks, Qwen 3B,
same prompts and scoring as the original battery.
- aac_wav (new): the Arm A MP4's own soundtrack decoded back to WAV
  (arm_a2/, ffmpeg pcm_s16le, tail trimmed to exactly 480000 frames; AAC
  priming stays in the waveform), fed through the image + audio path.
- raw_wav: the original WAV, image + audio path. aac_wav vs raw_wav
  isolates the codec, one door for both.
- aac_mp4: the original Arm A, video path. aac_mp4 vs aac_wav isolates
  the input path, same codec damage.
Provenance: provenance.md in this folder (SHA-256 for all 22 stimulus
files plus the still; the silence-duplication note for the codec,
patching, and tracing-2 stimulus sets is recorded there, per the ruling).
Results: codec2_qwen_results.csv.
