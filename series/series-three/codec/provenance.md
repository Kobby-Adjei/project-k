# Codec-control stimulus provenance (Nov 3, 2026)

Every stimulus file in the codec-control battery, its origin, and its SHA-256.
Built by build_battery.py (sounds and Arm A) and by the Handover Correction
step 4 pass (Arm A2). All files 10.000 s, 48000 Hz, 16-bit PCM unless noted.

## sounds/ (Arm B masters, raw WAV)
- codec_burst_quiet_raw.wav: 0ec5a2146b380645c5326c7292907d7352367b51da6166e2200884ca03803641
- codec_clicks_quiet_raw.wav: cc183e8131a439a4c289135cb317c8f06fdce3c78dc509e604edca1be12a5d4a
- codec_silence_raw.wav: 90ee494bd12323fb5b9c88046ee34e51d9ddc5e4eb8a9471ff937ed84436570b
- codec_speech_raw.wav: 1b4b8e3dd006c920250dcdfb3464a502b46b3a76d7a648958eb8ec28f9c0d833
- codec_tone_quiet_raw.wav: d375550a0d04ac7602225b444d03a0873c31d1e719f8db86a201c6ded612e67d
- codec_tone_ref_raw.wav: 5ef2cb98c09150f4b42130037f2d050b1dcb395388775de52a88dd10b3706fd9
- codec_tone_under_bed_raw.wav: d6a25896ce5852975d174a9f213be18513725c1f17fd47ebc4d8b2b28e41a94d

## arm_a/ (original Arm A: MP4, still picture + AAC via build_swap.py's exact mux recipe, no fixed bitrate set)
- codec_burst_quiet_aac.mp4: 5fd86adc6ff09469556d66074ad5cad07354039e8b49fc2db962f54274cc0b2f
- codec_clicks_quiet_aac.mp4: 64d80149e629faf19a7fdf7c5f4e1345dc019ba9279bd7c50280504e2f46b1a8
- codec_silence_aac.mp4: 2f5aa2bf58c44c6ae65de02d2aa2a643cdf8842b146908e17cffebcce1feaaee
- codec_speech_aac.mp4: 386dc8019c9bd0d2c2a0582ee0bd94ece6dd4b200f7723b5c2bf908d360b3d2d
- codec_tone_quiet_aac.mp4: df98b519c6a84e5d18ffd336d050a8d09f553ca41ac2e7790e3b265160f10de1
- codec_tone_ref_aac.mp4: faa6762fe33460c23fbd14e943b31fecf9e1f3e06d32d77b301ffc1ba0d16916
- codec_tone_under_bed_aac.mp4: f730d6cbd3efa95088adb33cbb0f7a5d18306127e111295c813fad922d2a123a

## arm_a2/ (Handover Correction step 4: the Arm A MP4s' audio decoded back to WAV,
ffmpeg pcm_s16le, tail trimmed to exactly 480000 frames; the AAC round trip's
priming delay stays in the waveform, as it does inside the MP4 itself)
- codec_burst_quiet_aac_decoded.wav: 7af7a66503263fd1aa72f2cb6eccd06122c1d3a778e0061fa7f8042a804aca66
- codec_clicks_quiet_aac_decoded.wav: 21abfd37146ea751051d6f5fd6aa0038459ef1bcd490b3fd83b319777eb350a5
- codec_silence_aac_decoded.wav: 90ee494bd12323fb5b9c88046ee34e51d9ddc5e4eb8a9471ff937ed84436570b
- codec_speech_aac_decoded.wav: 2a842b8747acafe0884d0cbaa793e3bd9b980b23eeee0281224cdf9b87160a30
- codec_tone_quiet_aac_decoded.wav: aea90d150dc3f3e43957f8c3814e98e668c44799b57cec3c279efb3fc1dd9e96
- codec_tone_ref_aac_decoded.wav: 92583f795feccbc5782782b772c6863ddf74779f493a9222730fcd41765e5477
- codec_tone_under_bed_aac_decoded.wav: 93244b4c668c61e61a423cc24eb6bff095cec3f5d976107a390db74ab4f8a05b

## codec_still.png (the swap still, both arms): 7ff12a63024bc263368121dd79d8674a35e6b844e59a6a7faa3e5aadcecb89fe

## Silence duplication, noted in the same pass (per the ruling)

- Codec set: one silence master; silence is digital zero, so its copies are
  identical by construction. No independence is claimed across its repeats.
- Patching set: the 8 silence clips are identical digital silence; all 8
  score the same baseline (0.3257) for that reason. Recipients paired against
  distinct tones are distinct runs; the silence donors are one stimulus.
- Tracing round 2 set: the 360 silence clips carry only 60 unique waveform
  hashes (per the council's audit of tracing2_manifest.csv), and 97 of the
  120 held-out silence clips exactly match a training waveform. The probe
  split was by clip, not by waveform identity, so the silence class leaked
  across the split. The tone and speech classes vary by construction
  (pitch, onset, level). Step 5's property probes must split by sound
  identity, on de-duplicated stimuli. This concern is logged here and in
  the Series Three record; round 2's read already narrows its claim to
  linear decodability of sound-vs-silence.
