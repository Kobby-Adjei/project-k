# Kill test: tagger hint (sound turned into text) on Pegasus — design frozen before any run
Date frozen: Nov 12, 2026, before the run. Kobby's go: "Run the tagger kill test" (Nov 11 evening decision, tapped again Nov 12). Council's test form (Claude, Nov 11, 2026): tagger alone vs model alone vs model + hint, same clips, silence control, wrong-sound lures, bar frozen first; the hint path must beat the model alone AND do no worse than the tagger alone on silence and lures.
Model under test: Pegasus 1.6, Twelve Labs Analyze, temperature 0, via custom.twelvelabs-temp4.

## The tagger
CLAP (laion/clap-htsat-unfused), a pretrained audio-text model, used as-is, zero-shot, no training by us. It scores the clip's full audio track against four candidate phrases, frozen: "running water", "a vacuum cleaner", "applause and clapping", "silence". The tagger's label for a clip is its top-scoring candidate. Mapped names: water, vacuum, applause, silence.
The tagger hears the full audio track of the same mp4 the model gets, extracted at 16 kHz mono, loudness untouched.

## Clips (the same 4 as the Path A kill test)
water = pilot2/swap/swap_water.mp4; vacuum = pilot2/swap/swap_vacuum.mp4; applause = docs-sounds-check/docs_applause_still.mp4; silence = pilot2/swap/swap_silence.mp4.

## Arms
- T (tagger alone): the tagger's top label per clip. Scored: top label equals the true sound (silence counts when the top label is "silence").
- M (model alone): REUSED from the frozen Path A kill test on Pegasus (same clips, same standard prompt P2 "Describe what happens in this video, including any sounds that are audible.", temperature 0): standard arm scored 0 of 3, silence clean. No new calls; the reuse is stated wherever the result is reported.
- H (model + true hint): the standard prompt with one line added before it: "Sounds detected: {tagger's label for this clip, in plain words}." For a silence top label the line reads "Sounds detected: no distinct sound." Full prompt per clip = the hint line, a blank line, then P2 verbatim. 4 calls (3 sounds + silence).
- L (model + wrong hint, the lure arm): same construction with a deliberately wrong label. water gets "a vacuum cleaner"; vacuum gets "applause and clapping"; applause gets "running water"; silence gets "running water". 4 calls. A lure answer that names the hinted wrong sound counts as a blind copy.

## Bar, frozen before the run
The hint path SURVIVES only if ALL hold:
1. H names the correct sound in at least 2 of 3 sound clips (M scored 0 of 3).
2. H on silence names no sound.
3. Blind copies in L are at most 1 of 4, and the silence lure (silence + "running water" hint) must NOT name water; if it does, the path fails on the spot regardless of the count.
If the tagger itself misses a sound in arm T, H missing that same sound is recorded as the tagger's miss carried through, and condition 1 is also read against the sounds the tagger got right (both readings reported).
Scoring of model answers: the same synonym rules as the earlier kill tests (water/water tap/dripping etc; vacuum/hoover; applause/clapping/claps; vague "noise" earns nothing).
Scope: one closed model, 3 sounds, 1 repeat per cell. A pass is a pilot, per the council: held-out clips and new sound types are required before anything is called a public solution. 8 new Pegasus calls in total.
