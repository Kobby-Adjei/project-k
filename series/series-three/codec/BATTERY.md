# Codec-loss control battery

Built by build_battery.py. Seven sounds, each exactly 10.000 s. WAV masters are 16-bit PCM, mono, 48000 Hz (the swap raw sample rate); the one exception is speech, a byte-for-byte copy of pilot2/swap/work/a_speech_raw.wav (dual-mono stereo, unchanged), so the speech control is the swap study's own file. Every RMS figure below, for WAV and MP4 alike, is measured by decoding the file to mono through the one identical ffmpeg chain, so the A/B deltas are same-chain comparisons.

Arm A MP4s use pilot2/swap/build_swap.py's mux recipe exactly: the same still (codec_still.png, a copy of the swap still), looped at 25 fps, libx264 yuv420p, native AAC at 48000 Hz, shortest, 10 s. The recipe sets no audio bitrate flag, and the encoder's rate follows the content: these seven files measure between about 1.5k and 69k bps, and the speech Arm A lands at about 38k bps, matching the shipped swap speech file's 38009 bps on the same soundtrack. The recipe, not the 128k label in the plan, is what Arm A reproduces.

Levels by design: tone_ref burst RMS -20 dBFS over its 0.5 s window (measured below); tone_quiet is the same burst 20 dB lower; each click is scaled to -40 dBFS RMS over its own 25 ms extent; the noise burst is scaled to -40 dBFS RMS over its full 2 s window; the bed in tone_under_bed is the swap study's matched vacuum track (whole-file RMS about -20 dB) with the tone_quiet burst mixed in at t=2.0 s in the WAV domain; silence is digital zeros.

Measured event windows: tone_ref burst -20.0 dBFS; tone_quiet burst -40.0 dBFS; clicks, per-click 25 ms RMS [-40.0, -40.0, -40.0] dBFS, peak -25.13 dBFS; burst_quiet 2 s RMS -40.0 dBFS; tone_under_bed whole file RMS -21.84 dB, peak 0.834.

| Sound | WAV | MP4 (Arm A) | WAV RMS (dB) | Decoded MP4 RMS (dB) | A/B delta (dB) | AAC bit rate (bps) |
|---|---|---|---|---|---|---|
| tone_ref | sounds/codec_tone_ref_raw.wav | arm_a/codec_tone_ref_aac.mp4 | -33.01 | -33.02 | -0.01 | 5675 |
| tone_quiet | sounds/codec_tone_quiet_raw.wav | arm_a/codec_tone_quiet_aac.mp4 | -53.01 | -53.02 | -0.01 | 5818 |
| clicks_quiet | sounds/codec_clicks_quiet_raw.wav | arm_a/codec_clicks_quiet_aac.mp4 | -61.25 | -61.37 | -0.12 | 3299 |
| speech | sounds/codec_speech_raw.wav | arm_a/codec_speech_aac.mp4 | -15.74 | -15.74 | 0.0 | 38503 |
| tone_under_bed | sounds/codec_tone_under_bed_raw.wav | arm_a/codec_tone_under_bed_aac.mp4 | -21.84 | -21.84 | 0.0 | 69280 |
| burst_quiet | sounds/codec_burst_quiet_raw.wav | arm_a/codec_burst_quiet_aac.mp4 | -46.99 | -48.32 | -1.33 | 16103 |
| silence | sounds/codec_silence_raw.wav | arm_a/codec_silence_aac.mp4 | -120.0 | -120.0 | 0.0 | 1527 |

## Per-sound detail

### tone_ref
- WAV: sounds/codec_tone_ref_raw.wav (channels: 1, duration: 10.0 s)
- WAV SHA-256: 5ef2cb98c09150f4b42130037f2d050b1dcb395388775de52a88dd10b3706fd9
- WAV per-second RMS (dB): [-120.0, -120.0, -23.01, -120.0, -120.0, -120.0, -120.0, -120.0, -120.0, -120.0]
- MP4: arm_a/codec_tone_ref_aac.mp4 (duration: 10.0 s, audio: aac at 5675 bps)
- MP4 SHA-256: faa6762fe33460c23fbd14e943b31fecf9e1f3e06d32d77b301ffc1ba0d16916
- MP4 decoded per-second RMS (dB): [-120.0, -90.71, -23.02, -120.0, -120.0, -120.0, -120.0, -120.0, -120.0, -120.0]

### tone_quiet
- WAV: sounds/codec_tone_quiet_raw.wav (channels: 1, duration: 10.0 s)
- WAV SHA-256: d375550a0d04ac7602225b444d03a0873c31d1e719f8db86a201c6ded612e67d
- WAV per-second RMS (dB): [-120.0, -120.0, -43.01, -120.0, -120.0, -120.0, -120.0, -120.0, -120.0, -120.0]
- MP4: arm_a/codec_tone_quiet_aac.mp4 (duration: 10.0 s, audio: aac at 5818 bps)
- MP4 SHA-256: df98b519c6a84e5d18ffd336d050a8d09f553ca41ac2e7790e3b265160f10de1
- MP4 decoded per-second RMS (dB): [-120.0, -106.58, -43.02, -120.0, -120.0, -120.0, -120.0, -120.0, -120.0, -120.0]

### clicks_quiet
- WAV: sounds/codec_clicks_quiet_raw.wav (channels: 1, duration: 10.0 s)
- WAV SHA-256: cc183e8131a439a4c289135cb317c8f06fdce3c78dc509e604edca1be12a5d4a
- WAV per-second RMS (dB): [-120.0, -120.0, -56.02, -120.0, -56.02, -120.0, -56.02, -120.0, -120.0, -120.0]
- MP4: arm_a/codec_clicks_quiet_aac.mp4 (duration: 10.0 s, audio: aac at 3299 bps)
- MP4 SHA-256: 64d80149e629faf19a7fdf7c5f4e1345dc019ba9279bd7c50280504e2f46b1a8
- MP4 decoded per-second RMS (dB): [-120.0, -100.58, -56.13, -100.57, -56.13, -100.58, -56.14, -120.0, -120.0, -120.0]

### speech
- WAV: sounds/codec_speech_raw.wav (channels: 2, duration: 10.0 s)
- WAV SHA-256: 1b4b8e3dd006c920250dcdfb3464a502b46b3a76d7a648958eb8ec28f9c0d833
- WAV per-second RMS (dB): [-120.0, -120.0, -9.76, -9.15, -14.0, -120.0, -120.0, -120.0, -120.0, -120.0]
- MP4: arm_a/codec_speech_aac.mp4 (duration: 10.0 s, audio: aac at 38503 bps)
- MP4 SHA-256: 386dc8019c9bd0d2c2a0582ee0bd94ece6dd4b200f7723b5c2bf908d360b3d2d
- MP4 decoded per-second RMS (dB): [-120.0, -129.25, -9.77, -9.15, -14.01, -120.0, -120.0, -120.0, -120.0, -120.0]

### tone_under_bed
- WAV: sounds/codec_tone_under_bed_raw.wav (channels: 1, duration: 10.0 s)
- WAV SHA-256: d6a25896ce5852975d174a9f213be18513725c1f17fd47ebc4d8b2b28e41a94d
- WAV per-second RMS (dB): [-31.54, -24.57, -20.75, -29.08, -30.32, -26.63, -26.33, -24.53, -17.75, -15.87]
- MP4: arm_a/codec_tone_under_bed_aac.mp4 (duration: 10.0 s, audio: aac at 69280 bps)
- MP4 SHA-256: f730d6cbd3efa95088adb33cbb0f7a5d18306127e111295c813fad922d2a123a
- MP4 decoded per-second RMS (dB): [-31.54, -24.67, -20.79, -29.15, -30.36, -26.7, -26.4, -24.51, -17.77, -15.81]

### burst_quiet
- WAV: sounds/codec_burst_quiet_raw.wav (channels: 1, duration: 10.0 s)
- WAV SHA-256: 0ec5a2146b380645c5326c7292907d7352367b51da6166e2200884ca03803641
- WAV per-second RMS (dB): [-120.0, -120.0, -120.0, -40.06, -39.94, -120.0, -120.0, -120.0, -120.0, -120.0]
- MP4: arm_a/codec_burst_quiet_aac.mp4 (duration: 10.0 s, audio: aac at 16103 bps)
- MP4 SHA-256: 5fd86adc6ff09469556d66074ad5cad07354039e8b49fc2db962f54274cc0b2f
- MP4 decoded per-second RMS (dB): [-120.0, -120.0, -149.43, -41.48, -41.19, -100.79, -120.0, -120.0, -120.0, -120.0]

### silence
- WAV: sounds/codec_silence_raw.wav (channels: 1, duration: 10.0 s)
- WAV SHA-256: 90ee494bd12323fb5b9c88046ee34e51d9ddc5e4eb8a9471ff937ed84436570b
- WAV per-second RMS (dB): [-120.0, -120.0, -120.0, -120.0, -120.0, -120.0, -120.0, -120.0, -120.0, -120.0]
- MP4: arm_a/codec_silence_aac.mp4 (duration: 10.0 s, audio: aac at 1527 bps)
- MP4 SHA-256: 2f5aa2bf58c44c6ae65de02d2aa2a643cdf8842b146908e17cffebcce1feaaee
- MP4 decoded per-second RMS (dB): [-120.0, -120.0, -120.0, -120.0, -120.0, -120.0, -120.0, -120.0, -120.0, -120.0]
