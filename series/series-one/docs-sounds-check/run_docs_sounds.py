#!/usr/bin/env python3
"""Docs sounds check runner (Nov 6, 2026).

Asks Pegasus 1.6 about the Twelve Labs docs page's own three
example sounds (birds chirping, applause, fireworks), using
the same prompts and call shape as the earlier batteries:
- P3 hearing form, imported byte-identical from run_swap.py,
  on the still version, 3 repeats per sound.
- Yes/no form from the Series Two battery, 2 repeats per cell:
  own target on still, on gray, and on the shared muted
  control (swap_silence.mp4), plus one wrong-target lure on
  still per sound.
Temperature 0. POST /v1.3/analyze, video inline base64, auth
via the connector named in TL_CONNECTOR (this run:
custom.twelvelabs-temp4, the tests-only account).
Resume-safe: raw JSON per job under raw/, finished jobs with
a 200 answer are skipped. Stops the whole run on HTTP 422
(allowance gone, exit 22). Backoff retries on other errors.
Output: docs_sounds_results.csv.
"""
import base64, csv, json, os, sys, time
import urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT2 = os.path.join(HERE, "..", "pilot2")
sys.path.insert(0, os.path.join(PILOT2, "swap"))
sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
from dynamic_credentials import add_surrogate_to_request  # noqa: E402
import run_swap  # noqa: E402 (exact P3 prompt)

RAW = os.path.join(HERE, "raw")
os.makedirs(RAW, exist_ok=True)
OUT = os.path.join(HERE, "docs_sounds_results.csv")
API_URL = "https://api.twelvelabs.io/v1.3/analyze"
MODEL = "pegasus1.6"
DELAY = 3.0
CONNECTOR = os.environ.get("TL_CONNECTOR", "custom.twelvelabs-temp4")

P3 = run_swap.PROMPTS["P3"]
OWN = {"birds": "birds chirping", "applause": "applause",
       "fireworks": "fireworks"}
LURE = {"birds": "applause", "applause": "fireworks",
        "fireworks": "birds chirping"}
MUTED = os.path.join(PILOT2, "swap", "swap_silence.mp4")


def file_for(sound, version):
    if version == "muted":
        return MUTED
    return os.path.join(HERE, f"docs_{sound}_{version}.mp4")


def build_jobs():
    jobs = []  # (sound, version, target, prompt_id, prompt, repeat, path)
    for sound in ("birds", "applause", "fireworks"):
        for rep in range(1, 4):
            jobs.append((sound, "still", "", "P3", P3, rep,
                         file_for(sound, "still")))
        cells = [("still", OWN[sound]), ("gray", OWN[sound]),
                 ("muted", OWN[sound]), ("still", LURE[sound])]
        for version, target in cells:
            prompt = (f"Is there the sound of {target} in this clip? "
                      f"Answer yes or no.")
            for rep in range(1, 3):
                jobs.append((sound, version, target, "YN", prompt, rep,
                             file_for(sound, version)))
    return jobs


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args):
        raise RuntimeError("redirect refused, verify the endpoint")


def one_call(path, prompt):
    with open(path, "rb") as fh:
        file_bytes = fh.read()
    payload = {
        "model_name": MODEL,
        "video": {"type": "base64_string",
                  "base64_string": base64.b64encode(file_bytes).decode("ascii")},
        "prompt": prompt,
        "temperature": 0,
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(API_URL, data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    add_surrogate_to_request(request, CONNECTOR,
                             allowed_hosts=["api.twelvelabs.io"])
    opener = urllib.request.build_opener(_NoRedirect())
    try:
        with opener.open(request, timeout=600) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")


def data_field_of(body):
    try:
        obj = json.loads(body)
    except Exception:
        return None
    if isinstance(obj, dict):
        return obj.get("data")
    return None


def main():
    jobs = build_jobs()
    for _, _, _, _, _, _, path in jobs:
        if not os.path.exists(path):
            print(f"MISSING FILE: {path}", file=sys.stderr)
            return 1
    print(f"{len(jobs)} calls queued via {CONNECTOR}")
    done = 0
    for sound, version, target, pid, prompt, rep, path in jobs:
        tag = f"{sound}__{version}__{pid}__{target.replace(' ', '_')}__r{rep}"
        out = os.path.join(RAW, f"{tag}.json")
        if os.path.exists(out):
            try:
                rec = json.load(open(out))
                if rec.get("http_status") == 200 and rec.get("answer"):
                    done += 1
                    print(f"[{done}/{len(jobs)}] {tag}: already saved, skip")
                    continue
            except Exception:
                pass
        status, body = 0, ""
        for attempt in range(3):
            status, body = one_call(path, prompt)
            if status == 200 or status == 422:
                break
            wait = 20 * (2 ** attempt)
            print(f"  http {status}, backing off {wait}s", flush=True)
            time.sleep(wait)
        if status == 422:
            print(f"[{done + 1}/{len(jobs)}] {tag}: HTTP 422 - allowance gone, STOP")
            print(body[:300])
            return 22
        rec = {"sound": sound, "version": version, "target": target,
               "prompt_id": pid, "prompt": prompt, "repeat": rep,
               "file": os.path.basename(path), "http_status": status,
               "raw_body": body, "answer": data_field_of(body)}
        with open(out, "w") as f:
            json.dump(rec, f, indent=1)
        done += 1
        print(f"[{done}/{len(jobs)}] {tag}: HTTP {status} "
              f"answer={str(rec['answer'])[:70]!r}", flush=True)
        time.sleep(DELAY)
    rows = []
    for fn in sorted(os.listdir(RAW)):
        if fn.endswith(".json"):
            rows.append(json.load(open(os.path.join(RAW, fn))))
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sound", "version", "target", "prompt_id", "prompt",
                    "repeat", "file", "http_status", "answer"])
        for r in rows:
            w.writerow([r["sound"], r["version"], r["target"],
                        r["prompt_id"], r["prompt"], r["repeat"],
                        r["file"], r["http_status"], r["answer"]])
    print(f"DONE: {len(rows)} records, CSV written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
