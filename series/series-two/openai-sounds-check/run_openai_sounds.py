#!/usr/bin/env python3
"""OpenAI-via-OpenRouter non-speech sound check (Nov 11, 2026).

Same sound files as the Series Two battery and the Nov 6 docs sounds
check (RMS-matched WAVs), same two ask forms, so results sit beside
Pegasus and Qwen on identical material. Saves every raw response;
scoring is done later from these files, never from this runner's tail.
Credential: custom.openrouter connector surrogate (Authorization bearer).
"""
import base64, json, os, sys, time
import urllib.request, urllib.error

sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
from dynamic_credentials import add_surrogate_to_request, read_response_body

OUT = os.path.expanduser("~/workspace/project-k-pilot/openai-openrouter-test/raw")
MODELS = ["openai/gpt-audio", "openai/gpt-audio-mini"]
SOUNDS = {
    "birds":     ("~/workspace/project-k-pilot/docs-sounds-check/work/a_birds_matched.wav", "birds chirping"),
    "applause":  ("~/workspace/project-k-pilot/docs-sounds-check/work/a_applause_matched.wav", "applause, people clapping"),
    "fireworks": ("~/workspace/project-k-pilot/docs-sounds-check/work/a_fireworks_matched.wav", "fireworks"),
    "water":     ("~/workspace/project-k-pilot/pilot2/swap/work/a_water_matched.wav", "running water"),
    "vacuum":    ("~/workspace/project-k-pilot/pilot2/swap/work/a_vacuum_matched.wav", "a vacuum cleaner"),
    "beeps":     ("~/workspace/project-k-pilot/pilot2/swap/work/a_beeps_matched.wav", "electronic beeps"),
    "sizzle":    ("~/workspace/project-k-pilot/pilot2/swap/work/a_sizzle_matched.wav", "food sizzling in a pan"),
    "speech":    ("~/workspace/project-k-pilot/pilot2/swap/work/a_speech_matched.wav", "a person speaking"),
    "silence":   ("~/workspace/project-k-pilot/pilot2/swap/work/a_silence_raw.wav", "any sound at all"),
}
OPEN_Q = ("What sounds can you hear in this audio clip? Describe only what is audible. "
          "If the audio is silent, say 'silent'.")

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args):
        raise RuntimeError("redirect refused")

def call(model, wav_path, question):
    with open(os.path.expanduser(wav_path), "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": question},
            {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
        ]}],
    }
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    add_surrogate_to_request(req, "custom.openrouter", allowed_hosts=["openrouter.ai"])
    opener = urllib.request.build_opener(NoRedirect())
    with opener.open(req, timeout=180) as resp:
        return resp.status, read_response_body(resp).decode("utf-8", errors="replace")

def main():
    done = fail = 0
    for model in MODELS:
        mslug = model.split("/")[1]
        for sound, (path, label) in SOUNDS.items():
            for form, q in [("open", OPEN_Q),
                            ("presence", f"Is there a sound of {label} in this audio clip? Answer only yes or no.")]:
                name = f"{mslug}__{sound}__{form}.json"
                dest = os.path.join(OUT, name)
                if os.path.exists(dest):
                    done += 1
                    continue
                try:
                    status, body = call(model, path, q)
                    with open(dest, "w") as f:
                        f.write(body)
                    print(f"OK {status} {name}", flush=True)
                    done += 1
                except urllib.error.HTTPError as e:
                    body = e.read().decode("utf-8", errors="replace")
                    print(f"FAIL HTTP {e.code} {name}: {body[:200]}", flush=True)
                    if e.code in (401, 403):
                        print("STOP: credential rejected", flush=True)
                        return 1
                    fail += 1
                except Exception as e:
                    print(f"FAIL {name}: {e}", flush=True)
                    fail += 1
                time.sleep(1)
    print(f"done={done} fail={fail}", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
