# Receipts: the exact answer texts

Every text below is copied character for character from the stored raw
records of the runs. These are the texts screenshots get taken from.
Where a text is quoted, nothing has been trimmed, fixed, or cleaned up.
Model for all Pegasus answers: pegasus1.6, Twelve Labs POST /v1.3/analyze,
temperature 0. Run dates: Nov 1, 2026.

## Post 1, The Mute Test: the eggs recording, clean vs muted

Question form: the repaired form (gist form 2). Clip: the researcher's
own eggs recording (own-01), clean file with real sound, and the same
file with the sound switched off. Three repeats each. The clean and
muted texts are character identical in all three repeats; repeat 1 is
shown.

Clean file, answer:

```
{"seen": "A person is cooking scrambled eggs in a pot on a gas stove. They turn the stove knob to ignite the burner, then use a spatula to stir and fold the eggs. A plate of rice sits on the counter next to the stove. Towards the end, the person transfers portions of the cooked eggs onto the plate with the rice.", "heard": "none", "sound_start_s": null}
```

Muted file, answer:

```
{"seen": "A person is cooking scrambled eggs in a pot on a gas stove. They turn the stove knob to ignite the burner, then use a spatula to stir and fold the eggs. A plate of rice sits on the counter next to the stove. Towards the end, the person transfers portions of the cooked eggs onto the plate with the rice.", "heard": "none", "sound_start_s": null}
```

The file's real sound: stove knob ignition clicks near the start, a
long egg sizzle, plating at the end. Measured in the clean file.

## Post 2, The Swap: sounds-only answers on one still picture

Question form: the sounds-only form (gist form 4). Same still picture
in both files; only the soundtrack differs. Three repeats each, one
identical text per cell.

Speech soundtrack (a synthesized voice saying "The blue kettle boils at
seven forty two"), answer:

```
Between 2.16 seconds and 4.64 seconds, a speaker says, "The Blue Kettle boils at seven forty two." No other sounds are audible in the clip.
```

Sizzle soundtrack (the eggs recording's first ten seconds: stove clicks
and sizzle, raised to the same loudness as the speech file), answer:

```
The audio track for this video clip is silent.
```

## Post 3, The Bed: the three matched-level transcripts

Question form: the sounds-only form (gist form 4). Speech level held
constant across all three files; a bed laid on top, never renormalized.
Three repeats each, one identical text per cell. True sentence: "The
blue kettle boils at seven forty two" (8 words).

Speech alone (baseline), answer:

```
Between 2.16 seconds and 4.64 seconds, a speaker says, "The Blue Kettle boils at seven forty two." No other sounds are audible in the clip.
```

Word differences from the true sentence: 0.

Speech over a beeps bed, answer (word for word identical to baseline,
same time window):

```
Between 2.16 seconds and 4.64 seconds, a speaker says, "The Blue Kettle boils at seven forty two." No other sounds are audible in the clip.
```

Word differences: 0. The bed is never reported.

Speech over a vacuum bed, answer:

```
The audio contains a single speaker stating, "Canon boils at seven forty two," between approximately 2.24 seconds and 4.56 seconds. No other sounds are audible in the clip.
```

Word differences: 3. "The blue kettle" collapsed into the single wrong
word "Canon", at exactly the words the bed sits under. The bed runs
under the voice for the full ten seconds, and the answer still says no
other sounds are audible.

## Post 5, Two AIs One Trap: the jar clip, two models

Same clip (a cup and a glass rinsed under a running tap), same repaired
question (gist form 2).

Pegasus 1.6, clean file, answer:

```
{"seen": "A person is washing dishes in a kitchen sink. They rinse a small cup and a glass under running water, then place the glass into a purple dish rack on the counter. Finally, they pick up a sponge to continue cleaning.", "heard": "none", "sound_start_s": null}
```

Note what the answer does: it names the running water under what it
saw, and reports nothing under what it heard.

Qwen2.5-Omni-3B (Colab notebook, same question, clean run, both repeats
the same), the heard field as recorded:

```
"heard": "water running from the faucet", "sound_start_s": 0
```

The sound is really there. The start is rough: Qwen said 0, the true
start is near 5 seconds. Source of record: qwen_omni_results.csv from
the run, written up in the project research journey, Nov 1, 2026.
Supporting line from the same run: on the tone control, Qwen heard "a
beep" (start 2.42s, true 3.0s), where Pegasus, asked directly how many
beeps there were, answered NO AUDIO, 3 of 3.
