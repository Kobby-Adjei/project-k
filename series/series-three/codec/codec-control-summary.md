Series Three, test 1 (codec-loss control). Series named Nov 3, 2026; map at ../series-three-inside-tests/README.md.

# Codec-loss control: read (Qwen 3B complete; MiniCPM pending)

Question (Kobby's, from the learning chat): if the file's lossy compression drops a quiet
sound before the model hears it, some "model misses" could be the file's fault. This control
feeds the same sounds two ways: Arm A inside an MP4 with the exact Pilot Two swap encode,
Arm B as the untouched WAV waveform, no codec anywhere. Same still picture, same prompts,
same settings. 56 asks per model (7 sounds x 2 arms x 2 forms x 2 repeats), feed proof with
hashes and hard stops throughout.

## File-level findings (before any model)

Measured through one identical decode chain on both sides (BATTERY.md):
- Most sounds pass through the AAC round trip almost untouched at the meter: whole-file
  RMS deltas 0.00 to -0.12 dB.
- The quiet broadband burst is the exception: -1.33 dB whole-file, its own seconds down
  about 1.3 dB, energy smeared into neighboring seconds. The codec measurably changes
  that one sound in the file itself.
- The swap encode sets no fixed bitrate: speech lands near 38 kbps, not 128k. (The 128k
  figure belongs to the Pilot Two clip cuts.) The control's file arm is therefore harder
  than the plan first stated.

## Qwen2.5-Omni-3B (complete Nov 3, 2026; free lane, L4; check exit 0; 56/56 asks;
CSV codec_qwen_results.csv, 56 rows, local copy in this folder)

Controls behaved. Silence: "silent" in the open form in both arms, No in both arms; no
invention anywhere. Speech, the denial control: open form "silent" in BOTH arms; the
reporting loss from Series Two reproduces exactly with the untouched wave. The yes score
underneath, though, nearly triples on the raw feed (0.156 AAC vs 0.464 raw) while the
spoken word stays No in both.

Loud tone (-20 dBFS): open form "silent" in BOTH arms. The loss is past the file.
Yes/no sits in the coin-flip zone (Yes at 0.443 in the file arm, No at 0.415 raw);
read the scores, not the words, for this model near 0.5.

Quiet sounds (-40 dBFS), where the codec hypothesis lives:
- Quiet tone: open form "silent" in the file arm, "beep" in the raw arm, both repeats.
  Rescued by the raw feed.
- Quiet clicks: open form "silent" in the file arm, detected in the raw arm ("a single
  beep"). Yes/no Yes in both arms, stronger raw (0.567 vs 0.460).
- Quiet burst: detected in the open form in both arms under wrong names (spray / gunshots).
  Yes/no for "sizzling" flips to Yes only in the raw arm (0.482 vs 0.407 No). This is the
  same sound the file measurement showed AAC damaging.
- Quiet tone under a loud vacuum bed (masking): missed in BOTH arms, yes scores about
  0.17 in both. Not a codec story; the bed swamps the tone inside the model either way.

## Verdict so far (one model)

The codec is mostly innocent. The headline Series One/Two losses (speech denial, sounds
unreported in the open form) happen with the untouched wave and live in the model, in the
recognition or the reporting. Compression has a real but small effect at the quiet level
tested: the weakest signals arrive stronger uncompressed, and in two of three quiet cases
the raw feed changed what the model reported. At normal levels, nothing moved.

## MiniCPM-o 2.6: PENDING

Run parked Nov 3, 2026: the paid account's Google verification offered only a passkey,
which cannot be completed from the remote browser (the Nov 2 pattern). Session left
untouched; a fresh attempt after a cool-down is the remedy. When it lands, append its
read here: MiniCPM exposes no yes score, so its evidence is the open answers and the
yes/no words, and the question is whether its quiet misses move between arms.

## Step 4 rerun (The Handover Correction, Nov 3, 2026): the confound, fixed

The council was right about the design: the original Arm A rode the video
path and Arm B the image + audio path, so the contrast mixed the codec with
the door. The rerun (codec2_qwen_results.csv, 84 asks, Qwen 3B) carries
three arms: aac_wav (the Arm A soundtrack decoded back to WAV, fed through
the image + audio door), raw_wav (the original WAV, same door), and aac_mp4
(the original video door). aac_wav vs raw_wav isolates the codec;
aac_mp4 vs aac_wav isolates the door. Provenance: provenance.md.

What survives, cleanly now:
- Speech is denied identically through one door (open "silent" in both;
  yes scores 0.4375 AAC vs 0.4641 raw). The denial is not the codec.
- The loud tone is unreported in the open form in both arms. Not the codec.
- The quiet tone under the vacuum bed is missed in both. Masking is not
  the codec.
- The codec's real signature stands, in the open form: through the same
  door, the quiet tone is named "beep" raw and "silent" after the AAC
  round trip, and the quiet clicks are caught raw and "silent" after it.
  Their yes scores barely move (0.37 both; 0.57 both): the codec nibbles
  at naming the quietest sounds, exactly the original read's claim.

What the rerun corrects:
- burst_quiet's original yes/no flip (No through the file, Yes raw) was
  mostly the door: through one door, both arms answer Yes (0.5045 vs
  0.4824). The file-level damage measured for the burst is real; its
  original detection flip is not a codec effect.
- The door itself is a factor: the same codec-damaged waveform scores up
  to ~0.28 apart through the two doors (speech yes score 0.1560 via video
  vs 0.4375 via image + audio), with no consistent direction. Any
  file-based claim in this project must now name its door.
