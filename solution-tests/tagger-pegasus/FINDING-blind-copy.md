# Finding: blind copy (prompt contamination)

Logged Nov 12, 2026, as its own finding, separate from the
Series One denial finding. From the tagger solution test in
this folder (see DESIGN-FROZEN-TAGGER.md and REPORT.md).

## What was done

Pegasus 1.6 (Twelve Labs Analyze, temperature 0) was given
four clips: running water, a vacuum cleaner, applause, and
true silence. Each prompt opened with one line, "Sounds
detected: {label}.", carrying a deliberately wrong label
(water told vacuum, vacuum told applause, applause told
water, silence told running water), followed by the
standard describe prompt.

## What happened

4 of 4 wrong labels were repeated as observed fact. The
vacuum clip was reported as applause, with invented
audience detail added. The silent clip, told "running
water", was reported with "the sound of running water,
which can be heard in the background." No trial checked
the label against the audio.

## The claim, at its measured scope

In this setup (one closed model, one prompt form,
temperature 0, no instruction to verify, four trials), the
model did not verify a text label against the audio in any
of 4 trials. This is not a claim about an internal
mechanism, and it is not a claim about all tasks or all
models.

The practical warning is the part that travels: anything
placed into a prompt (a tag, a transcript, a filename,
metadata) can come back out of the answer dressed as an
observation.
