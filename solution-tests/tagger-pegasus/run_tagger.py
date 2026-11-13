#!/usr/bin/env python3
"""Tagger arm (T) of the tagger kill test: CLAP zero-shot on the 4 clips.
Design: DESIGN-FROZEN-TAGGER.md. Candidates frozen; top score = the label.
Direct processor+model calls (the pipeline wrapper in transformers 5.x
rejects dict inputs). The processor is told the true 16 kHz rate and
resamples to CLAP's 48 kHz itself.
"""
import json, os
import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import ClapModel, ClapProcessor

HERE = os.path.dirname(os.path.abspath(__file__))
CLIPS = [("water", "tagger-clips/water_full.wav"),
         ("vacuum", "tagger-clips/vacuum_full.wav"),
         ("applause", "tagger-clips/applause_full.wav"),
         ("silence", "tagger-clips/silence_full.wav")]
CANDIDATES = ["running water", "a vacuum cleaner",
              "applause and clapping", "silence"]
MAP = {"running water": "water", "a vacuum cleaner": "vacuum",
       "applause and clapping": "applause", "silence": "silence"}

model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
model.eval()

out = {}
for name, rel in CLIPS:
    wave, sr = sf.read(os.path.join(HERE, rel), dtype="float32",
                       always_2d=False)
    if wave.ndim > 1:
        wave = wave[:, 0]
    # CLAP's extractor takes only its training rate (48 kHz) and does
    # not resample in transformers 5.x; resample 16k -> 48k here.
    if sr != 48000:
        wave = torchaudio.functional.resample(
            torch.from_numpy(wave), sr, 48000).numpy().astype(np.float32)
        sr = 48000
    inputs = processor(text=CANDIDATES, audio=[wave], sampling_rate=sr,
                       return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits_per_audio[0]
    probs = torch.softmax(logits, dim=-1).tolist()
    ranked = sorted(zip(CANDIDATES, probs), key=lambda x: -x[1])
    out[name] = {"file": rel,
                 "scores": {MAP[k]: round(float(v), 4)
                            for k, v in ranked},
                 "top_label": MAP[ranked[0][0]],
                 "top_phrase": ranked[0][0]}
    print(name, "->", out[name]["top_label"], out[name]["scores"],
          flush=True)
with open(os.path.join(HERE, "tagger_results.json"), "w") as fh:
    json.dump(out, fh, indent=1)
print("saved tagger_results.json")
