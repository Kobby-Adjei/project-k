# Next models: the test plan (nothing run yet)

Kobby's direction, Nov 1, 2026: test other models with the same
battery that produced Series One. This file is the plan only. No model
below has been run, and no result is claimed for any of them.

## The battery is fixed

Every model gets the same three tests, the same files, and the same
questions as the first model. One question for every model inside a
comparison, or the comparison proves nothing.

1. **The real clips.** The repaired question form (gist form 2), word
   for word, on the same three clips: the vacuum clip, the jar clip,
   and his own eggs recording, clean and muted, repeats as in Series
   One. Scored on what the answer puts under heard, and whether clean
   and muted answers still match letter for letter.
2. **The swap matrix.** The same seven built files: one still picture,
   seven loudness matched soundtracks. All three question forms (gist
   forms 2, 3, and 4). Scored per soundtrack: reported, missed, or
   named wrong.
3. **The bed test.** The same three matched-level files: speech alone,
   speech over beeps, speech over the vacuum, with the speech level
   held constant. Scored by word differences from the true sentence,
   and by whether either bed is ever reported.

Settings: temperature 0 where the service allows it. Where a service
does not allow a setting, or feeds the picture differently, that
difference is written into the record before any result is read.

## Candidates, with access notes to verify at run time

Access, pricing, and model names change. Nothing below asserts what is
true today; each line is a check to run on the day, before any call.

- **Gemini (Google).** Video understanding models, reached through
  Google AI Studio or the Gemini API. Verify at run time: which model
  version takes video with sound, how the file is uploaded, and what
  the current limits are. Notebook and Colab work runs on his
  kobbytti@gmail.com account, which carries the paid Colab plan
  (moved there Nov 1, 2026); the school account's Drive keeps the
  Project K documents.
- **GPT-class video-capable models (OpenAI).** Reached through the
  OpenAI API. Verify at run time: whether the current models accept
  video input with audio at all, and in what form (frames, files, or
  streams). If a model takes frames only, it is recorded as a
  picture-only input and is not scored on hearing.
- **Claude (Anthropic).** Verify at run time: whether video input is
  supported through the API, and with what sound handling. If only
  frames are accepted, same picture-only rule as above.
- **Open omni models on Colab.** Qwen2.5-Omni-7B first, as the step up
  from the 3B already run, then the other open omni models in the
  running order below. Same notebook pattern as the 3B run: sanity
  check after load, raw output always saved, failures printed loudly,
  never folded into empty answers. Verify at run time: GPU memory fit
  and load errors, the exact two failure modes the 3B run already
  taught.

## Open model running order (set Nov 1, 2026)

Only models that take sound and picture together can sit this exam.
Picture-only models are not on this list, however strong they are.

After the Qwen2.5-Omni-7B (already built, running Nov 1):

1. **MiniCPM-o 2.6** (about 8B). Built for live seeing and hearing
   together; the most used open omni model in its class. Fits the
   free chip class.
   RUN Nov 2, 2026: heard water on the jar clip and the swap water
   track, none on every other Part A cell, bed test void on a
   failed baseline. Full results: pilot2/minicpm-o/summary.md.
2. **OLA** (7B). Trained to line up what is heard with what is seen,
   which is the exact failure zone this battery probes.
3. **Baichuan-Omni 1.5** (7B). Small, and scored well for its size on
   a public audio-video test (SONIC-O1).
4. **Gemma 4, 12B** (Google, open weights). The different design: no
   sound encoder; raw sound is projected straight into the word
   space. Its failure shape should differ, which is the information.
   May need 8-bit loading on the free chip.
5. **VideoLLaMA2** and **video-SALMONN 2** (7B class). Older builds
   with a sound encoder bolted onto a video model; their sound path
   is separate, so they can fail in a different place.
6. **VITA-1.5** (7B class). A fourth family as a baseline.
7. **Qwen3-Omni-30B-A3B.** The strongest open model on the public
   audio-video test, and the summit of this list. Too heavy for the
   free chip; it waits for a bigger chip or heavy compression.
8. **Uni-MoE-2.0-Omni** (33B). Same weight problem as the 30B, lower
   priority.

Landscape note: the public SONIC-O1 benchmark tests this same class
of models on audio-video clips, and its weakest column for every
model, open and closed, is placing things in time. This battery is
smaller and narrower on purpose: it asks where the sound dies, which
that benchmark does not.

## Rules

- **His material first.** His own recordings, the generated still, and
  the synthetic voice and tones carry no licensing drag and go to every
  model. Dataset excerpts are used only where the battery needs them
  and are never republished without the licensing check.
- **Identical files.** Every model is fed the same built files, finger
  printed in the gist, so a difference in answers cannot be blamed on
  a difference in inputs.
- **Compare only what was fed.** A model is only ever compared on what
  it actually received. Frame rate, picture size, and any sound
  handling a service applies are recorded and stated beside its
  results, the same honest note the Qwen run carries.
- **Same scoring.** Heard or not, named right or wrong, clean against
  muted, word differences on the bed test. The scoring does not bend
  per model.
- **Nothing is a ranking.** The output of each run is a behavior
  report on that model, through that door, on those dates, in the
  Series One shape: what it shows, and what it does not prove.
