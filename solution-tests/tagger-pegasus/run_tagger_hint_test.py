#!/usr/bin/env python3
"""Tagger kill test: Pegasus hint (H) and wrong-hint lure (L) arms.
Design in DESIGN-FROZEN-TAGGER.md. Hints for H come from tagger_results.json
(the tagger's own top labels). Resume-safe; stops on HTTP 422.
"""
import base64, csv, json, os, sys, time
import urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
from dynamic_credentials import add_surrogate_to_request  # noqa: E402

RAW = os.path.join(HERE, "raw_tagger")
os.makedirs(RAW, exist_ok=True)
OUT = os.path.join(HERE, "tagger_hint_results.csv")
API_URL = "https://api.twelvelabs.io/v1.3/analyze"
MODEL = "pegasus1.6"
DELAY = 3.0
CONNECTOR = os.environ.get("TL_CONNECTOR", "custom.twelvelabs-temp4")

P2 = "Describe what happens in this video, including any sounds that are audible."
CLIPS = {
    "water": os.path.join(ROOT, "pilot2", "swap", "swap_water.mp4"),
    "vacuum": os.path.join(ROOT, "pilot2", "swap", "swap_vacuum.mp4"),
    "applause": os.path.join(ROOT, "docs-sounds-check",
                             "docs_applause_still.mp4"),
    "silence": os.path.join(ROOT, "pilot2", "swap", "swap_silence.mp4"),
}
PHRASE = {"water": "running water", "vacuum": "a vacuum cleaner",
          "applause": "applause and clapping", "silence": None}
WRONG = {"water": "a vacuum cleaner", "vacuum": "applause and clapping",
         "applause": "running water", "silence": "running water"}


def hint_prompt(phrase):
    if phrase is None:
        line = "Sounds detected: no distinct sound."
    else:
        line = f"Sounds detected: {phrase}."
    return line + "\n\n" + P2


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args):
        raise RuntimeError("redirect refused, verify the endpoint")


def one_call(path, prompt):
    with open(path, "rb") as fh:
        file_bytes = fh.read()
    payload = {"model_name": MODEL,
               "video": {"type": "base64_string",
                         "base64_string": base64.b64encode(
                             file_bytes).decode("ascii")},
               "prompt": prompt, "temperature": 0, "stream": False}
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(API_URL, data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    add_surrogate_to_request(request, CONNECTOR,
                             allowed_hosts=["api.twelvelabs.io"])
    opener = urllib.request.build_opener(_NoRedirect())
    try:
        with opener.open(request, timeout=600) as resp:
            return resp.status, resp.read().decode("utf-8",
                                                   errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")


def data_field_of(body):
    try:
        obj = json.loads(body)
    except Exception:
        return None
    return obj.get("data") if isinstance(obj, dict) else None


def main():
    tagger = json.load(open(os.path.join(HERE, "tagger_results.json")))
    jobs = []
    for sound in ("water", "vacuum", "applause", "silence"):
        top = tagger[sound]["top_label"]
        jobs.append((sound, "H", tagger[sound]["top_phrase"]
                     if top != "silence" else None))
        jobs.append((sound, "L", WRONG[sound]))
    rows = []
    for sound, arm, phrase in jobs:
        raw_path = os.path.join(RAW, f"{sound}_{arm}.json")
        prompt = hint_prompt(phrase)
        if os.path.exists(raw_path):
            body = open(raw_path).read()
            status = 200
        else:
            status, body = one_call(CLIPS[sound], prompt)
            with open(raw_path, "w") as fh:
                fh.write(body)
            print(f"{sound} {arm}: HTTP {status}", flush=True)
            if status == 422:
                print("STOP: HTTP 422 (allowance or validation). "
                      "Not retrying.")
                break
            time.sleep(DELAY)
        answer = data_field_of(body)
        rows.append({"sound": sound, "arm": arm,
                     "hint": phrase if phrase is not None
                     else "no distinct sound",
                     "http": status, "prompt": prompt,
                     "answer": answer if answer is not None else ""})
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["sound", "arm", "hint",
                                           "http", "prompt", "answer"])
        w.writeheader()
        w.writerows(rows)
    print(f"rows saved: {len(rows)} -> {OUT}")


if __name__ == "__main__":
    main()
