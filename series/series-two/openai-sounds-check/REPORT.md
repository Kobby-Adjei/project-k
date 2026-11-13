# OpenAI models, non-speech sound check (run Nov 11, 2026)

Question: do OpenAI's audio models catch non-speech sounds, on the
same files where Pegasus denied them? Route: OpenRouter
(custom.openrouter connector, Kobby's key), models openai/gpt-audio
and openai/gpt-audio-mini, the only two OpenAI models on OpenRouter
that accept audio input. Temperature 0, one run per cell.
Files: the battery's own RMS-matched WAVs (birds, applause,
fireworks from the docs sounds check; water, vacuum, beeps, sizzle,
speech from the swap set; silence raw as control). Two asks per
sound: open form ("What sounds can you hear in this audio clip?
Describe only what is audible. If the audio is silent, say
'silent'.") and presence ("Is there a sound of X in this audio
clip? Answer only yes or no."). 36 of 36 calls HTTP 200.
Raw answers: ~/workspace/project-k-pilot/openai-openrouter-test/raw/
All scoring below recomputed from those files.

Caveat: these are audio-only models, given the sound alone with no
video track. The ask forms are the battery's sounds-only forms.
Presence "yes" answers are generous: gpt-audio said yes to vacuum
and sizzle while describing different sounds in open form, so the
open form is the stricter read.

## gpt-audio (9 sounds)
Open form named the sound: birds (chirping, singing), applause
(clapping and applauding), water (dripping or trickling, splashing,
flowing), beeps (beeping tones). 4 of 7 non-speech sounds.
Missed: fireworks (heard "wind and possibly some objects being
moved or knocked around... thudding"), vacuum (heard "objects
clattering or being placed down, possibly dishes or metal items"),
sizzle (heard "metal clinking or rattling").
Presence: yes on 6 of 7 non-speech sounds (all but fireworks).
Speech control: presence yes, but the open form produced no
transcription at all, only a stall: "I will listen to the audio
and describe what is audible. Let me analyze it."
Silence control: failed oddly. Both asks were answered as if no
clip had been attached: "Please provide the audio clip, and I will
listen to it and describe the sounds that are audible."

## gpt-audio-mini (9 sounds)
Open form named the sound: birds, applause. 2 of 7 non-speech.
Its main failure is refusal: on fireworks, water, beeps, sizzle it
answered "I'm sorry, but I can't identify or describe the sounds
from the audio clip" in open form, and refused most presence asks
the same way. It refused the SPEECH clip outright in both forms
("I can't listen to or analyze audio") despite being an audio
model, and refused the silence clip too. Vacuum got a description
(clattering, scraping, "a loud crash") but not the name.

## Read
The flat denial is not universal. gpt-audio names birds, applause,
water, and beeps freely, where Pegasus named none of its docs
sounds in 21 tries. But the family is not clean: both OpenAI
models miss fireworks and vacuum in open form, gpt-audio renames
vacuum and sizzle as kitchen clatter, gpt-audio cannot transcribe
on the open ask and treats silence as a missing attachment, and
the mini model's first move on most sounds is to deny it can hear
at all. Failure kinds so far: Pegasus denies, MiniCPM loses the
report, gpt-audio renames and stalls, gpt-audio-mini refuses.
