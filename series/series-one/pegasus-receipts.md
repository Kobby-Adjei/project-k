# Where did the sound go? Reproducibility pack

Pilot tests of audio reporting by Pegasus 1.6 through Twelve Labs
Analyze, run Oct 31 to Nov 1, 2026 (America/Detroit). Everything needed
to repeat the tests is on this page: the exact request shape, every
question form word for word, the call counts, and SHA-256 fingerprints
of the evidence media. If any of this is wrong, the fingerprints and the
exact questions are here so the error can be found and shown.

## Service, model, and request shape

Service: Twelve Labs API, `POST https://api.twelvelabs.io/v1.3/analyze`
Model: `pegasus1.6`
Temperature: 0 on every call. Stream: false.

Inline door (the file travels inside the request):

```json
{
  "model_name": "pegasus1.6",
  "video": {"type": "base64_string", "base64_string": "<the media file, base64 encoded>"},
  "prompt": "<one of the question forms below, exactly>",
  "temperature": 0,
  "stream": false
}
```

Headers: `Content-Type: application/json`, and an authorization bearer
key held by the account owner. The key value is REDACTED here and is
never published. Any valid Twelve Labs API key on any account can run
the same requests.

Indexed door (the documented path, tested separately): an index was
created with models `[{"model_name": "marengo3.0", "model_options":
["visual", "audio"]}]`, so audio was declared at index creation. The
four condition files were uploaded as assets and indexed to ready, then
analyzed with the same endpoint, model, temperature, and repaired
question, with the video field set to
`{"type": "asset_id", "asset_id": "<the indexed asset>"}`.

The answer text is read from the response `data` field, stored exactly
as returned.

## Every question form, verbatim

Form 1, the original form (398 characters). Used in the smoke run, and
in the probe's speech control:

> You are given a short first-person video clip. Watch and listen to the whole clip.
> Respond with ONLY a JSON object in exactly this form, no other text: {"seen": "<the main things happening that you can SEE, short>", "heard": "<the main things happening that you can HEAR, short; use an empty string if there is no sound>", "sound_start_s": <number: the second at which the main sound event starts>}

Known flaw, stated openly: this form invites an empty string for
silence and gives the start field no way to say "none", so its zeros
prove nothing on their own. That is why form 2 exists.

Form 2, the repaired form (444 characters). Used in the eggs run, the
repaired rerun, the indexed run, the swap run (as prompt P1), and every
Qwen ask:

> You are given a short first-person video clip. Watch and listen to the whole clip.
> Respond with ONLY a JSON object in exactly this form, no other text: {"seen": "<the main things happening that you can SEE, short>", "heard": "<the main things happening that you can HEAR, short; write the word none if you hear no sound at all>", "sound_start_s": <number: the second at which the main sound event starts, or the word null if there is no sound>}

Form 3, plain description (75 characters). Swap run prompt P2:

> Describe what happens in this video, including any sounds that are audible.

Form 4, the sounds-only form (143 characters). Swap run prompt P3, and
the only form used in the matched-level run:

> What sounds can you hear in this clip? Describe only what is audible, without guessing from the pictures. If the audio is silent, say 'silent'.

Form 5, the probe's direct speech question (129 characters):

> Listen to this clip. What sentence is spoken? Quote it exactly. If you cannot hear any audio at all, reply with exactly: NO AUDIO

Form 6, the probe's direct beep question (158 characters):

> Listen to this clip. How many beeps are there, and at what second does the first beep start? If you cannot hear any audio at all, reply with exactly: NO AUDIO

## Runs and call counts (Pegasus 1.6, all HTTP 200)

- Smoke run, Oct 31 to Nov 1, 2026: 2 clips (vacuum, jar) x 3 conditions
  (clean, muted, picture covered over the sound window) x 5 repeats,
  form 1. 30 calls.
- Control probe: speech file with form 5 x3 and form 1 x2, tone file
  with form 6 x3. 8 calls.
- Eggs run: the filmer's own recordings (2 clips clean and muted, 1
  clip clean only) x 3 repeats, form 2. 15 calls.
- Repaired rerun: vacuum and jar, clean and muted, x 5 repeats, form 2.
  20 calls.
- Indexed run: the same 4 files through the indexed door, x 5 repeats,
  form 2. 20 calls.
- Swap run: 7 soundtracks on one still picture x 3 question forms
  (forms 2, 3, 4) x 3 repeats. 63 calls.
- Matched-level run: 3 files (speech alone, speech over a vacuum bed,
  speech over a beeps bed) x form 4 x 3 repeats. 9 calls.
- Total Pegasus calls: 165.

Second model, feasibility only: Qwen2.5-Omni-3B, run in a Google Colab
notebook on a T4 GPU, form 2, 5 clips x 2 repeats, 10 asks in the clean
run. Its picture feed was downsampled (1 frame per second, capped
pixels), so it is not a controlled head to head and is never counted in
the 165. An earlier Qwen run hid an out of memory error behind empty
fields; those results were discarded and the run is excluded.

## Media and measurements

- Vacuum and jar clips: two EPIC-KITCHENS clips whose sound moments
  carry no matching visual label, human checked. Clean audio measured
  in the files sent: vacuum clip RMS -26.8 dB, jar clip RMS -39.7 dB.
  Muted versions are true digital silence (volume set to 0, measured
  RMS minus infinity). EPIC media: dataset terms apply.
- Own recordings: filmed by the researcher on Nov 1, 2026, first
  person, cooking eggs on a gas stove; private, never posted, filmed
  after the model's Oct 6, 2026 release. Clean audio measured: own-01
  RMS -38.3 dB with peaks near full volume.
- Swap files: one generated still picture (854x480, dark field with a
  light circle), 10 seconds each, first frame verified identical across
  all seven files. Soundtracks loudness matched to RMS about -20 dB,
  measured from the decoded finished files: silence (true digital
  silence), a synthesized spoken sentence ("The blue kettle boils at
  seven forty two", starting at 2.0s), three 1000 Hz beeps (at 3.0,
  4.5, 6.0s), running water, a vacuum, an egg sizzle with stove clicks,
  and the sentence mixed over the vacuum.
- Matched files: the same still picture and the same speech track at a
  held constant level (decoded speech RMS -20.00 dB in every file,
  never renormalized after a bed was added). Beds measured from bed-only
  reference files at the exact mix gains: vacuum RMS -20.71 dB, beeps
  RMS -20.00 dB. Note the fingerprint table below shows the matched
  baseline file is byte identical to the swap speech file, which is
  itself proof the baseline did not drift between runs.

## SHA-256 fingerprints of the evidence media

Paths are relative to the project pilot2 folder.

Run 1 condition files (EPIC media, dataset terms apply):

```
32fbd1791f64967f7e20b1099820a1f68fe2ac17f9fef32d610581dc1ea91c49  run1/conditions/P02_07_5__clean.mp4
17e32f4e215b5b3f3246d226f94816293e14a0657e33496efe415c7d7c685b7a  run1/conditions/P02_07_5__audio_muted.mp4
fb7fa6e15affa0c1a5ce1ba280ba00e9f0e2d4c76c0cd751297607c566bf1d2e  run1/conditions/P32_01_54__clean.mp4
ede1d5a51957b086dc64fc199b338bfa16946177012ad1dcedc8ca7c6b9d70b4  run1/conditions/P32_01_54__audio_muted.mp4
```

Own recordings (researcher's own media):

```
f31eecfe2fa06335d65890fffc61b665146fd64e5a232d4cfd5c82830f4444e5  own-recordings/own-01.mp4 (source recording)
06fdfd1c56a6589dc4412786e20efd5a2784b9207905d307a10172f3e35019df  run-eggs/conditions/own-01__clean.mp4 (as sent)
1214d7289664d1a55b473d3a49280f5907ef49069e0fd86d61f4a5657533fc22  run-eggs/conditions/own-01__audio_muted.mp4 (as sent)
```

Control probe files (synthetic):

```
df53d72f2c91529ed4cb1feeb00e436b89753101257317eeafb39c59bb7fc2ef  probe/probe_speech.mp4
773658ab32e6485c261e362af15bbc13c64f3ad2eab82f0c986e3a44651f9e51  probe/probe_tone.mp4
```

Swap files (generated still, owned and synthetic material):

```
a78522c561bce2fb960edda290f8926f36eabc932a81f74cbce4b223fa64abfa  swap/swap_silence.mp4
e9417b7b02dffe84587fd21b1c4c06c0d37920309b747504e8a64533815a49b5  swap/swap_speech.mp4
1e488e2999a5fc2e5f20a79d75da8f46f0633e6e9a727b54e6b3c8c6637c110e  swap/swap_beeps.mp4
3a94d02f214e89050a86c8cacb8352e9edc1ed94ecb26594a4baaacb0b3bf434  swap/swap_water.mp4
773b5b11fcf5e7f84e0b3c0c9a5c39bc724f75c2afe4e419f8256ee988861b76  swap/swap_vacuum.mp4
fae12f6b9623917d0332d2b5f4f3ff846b69ed8f909f05540a307ec6826f673d  swap/swap_sizzle.mp4
f610ed0b441f8dbe70573aef0f8ce1b5663f9970dbb03dd223d091ecb9a15fc0  swap/swap_mixed.mp4
```

Matched-level files:

```
e9417b7b02dffe84587fd21b1c4c06c0d37920309b747504e8a64533815a49b5  swap/matched/matched_speech.mp4
5ec86fe7d9ab31103b84a57e1a953dfac7b97fd8dc0a90eab44c52d191adcff6  swap/matched/matched_speech_vacuum.mp4
8b3ce7b3f64b240b622216b10cfb5cba682279784bd9b134ed0f42243a44ec77  swap/matched/matched_speech_beeps.mp4
```

## Scope: what this shows and what it does not show

Scope: Pegasus 1.6 via Twelve Labs Analyze, inline and indexed doors,
Oct 31 to Nov 1, 2026. Six question forms, temperature 0, every repeat
within a cell identical.

What the results show: on every clip and soundtrack tested, the model
reported no non-speech sound, under every question form, on both doors.
Answers to clean and muted files were character identical. A spoken
sentence was recovered and quoted. At matched loudness, beeps, running
water, a vacuum, and a sizzle drew "silent" or "none", and a sentence
laid over a vacuum bed came back with its opening words damaged by the
bed while the answer denied any other sound was audible.

What this does not show: anything about other models, other endpoints,
other model versions, or behavior after Nov 1, 2026. It does not locate
where inside the service the sound is lost; the bed test shows
non-speech sound can reach and bend the speech transcript, and that it
is then dropped from the report, and the claim stops there. The real
clips are few (honest count: independent clips, not repeats), and one
control sound (the beeps) is a synthetic tone, not a natural sound.

If I have this wrong, show me where, and I will post the correction.

Last updated Nov 1, 2026.
