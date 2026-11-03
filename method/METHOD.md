# Method

The rules every test in this repo runs under. They exist
because each one was earned by a failure.

## One question, every model

Inside a comparison, the question never changes. The same
files, the same prompt text, byte for byte, the same
decoding settings (temperature 0, greedy where the model
allows it). If a question has to change, the comparison
restarts. A result is only ever read against a baseline run
under the identical question.

## Prove the feed, per ask

Every ask prints its own proof from inside the session that
the file it received carries the sound it is supposed to
carry: duration, track level, nonzero share. If a sound
carrying track decodes missing, empty, suspiciously short,
or all zeros, the run stops loudly instead of padding the
track and continuing. Declared silence is the one exempt
case, verified as true zeros. This rule exists because an
early runner silently replaced a missing track with an
empty one, and every result downstream of that would have
been fiction.

## Baselines before beds

Any test that layers one sound over another is read only
through a question whose baseline, the same question on the
clean file, returned the expected answer. A failed baseline
voids the rows under it, and the void is reported as a
result, not hidden.

## Controls beside every claim

A yes means nothing without its muted control beside it.
Muted and blackout versions are built from the same master
files as the sound-on versions, and every file in a battery
carries a SHA-256 fingerprint so a reader can verify the
files are what the results claim they are.

## Stages are hypotheses

Findings are sorted into a working taxonomy of where a
sound can be lost: feed integrity, event recognition,
cross-modal interference, reporting, and temporal
grounding, with hallucination tracked separately. These are
hypothesized stages, used to organize measurements. A
measurement can show that a sound was lost and which
question forms lose it. It cannot show where inside a
closed system the loss happens, and no claim here says
otherwise.

## Levels are stated

The kitchen clips are quiet compared to the made files; the
quietest sits near -40 dB. Sound levels are measured and
recorded per file, and level caveats travel with the
results they affect.
