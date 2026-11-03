#!/usr/bin/env python3
"""Pegasus text version of the Series Two Test 3 yes/no battery.

Claude's Q4 (Nov 2, 2026 council): the open models get yes/no
scoring on the swap files in three versions; "Pegasus can't
expose probabilities, so use its yes/no text answers plus lure
questions." This runner asks the SAME 30 cells the open models
got in Test 3 (same files, same targets, same prompt form),
2 repeats each = 60 calls, through the same door as every
Pilot Two Pegasus run: POST /v1.3/analyze, pegasus1.6,
temperature 0, video inline base64, auth via the
custom.twelvelabs-run connector surrogate.

Files: sound_on_still = pilot2/swap/swap_<sound>.mp4;
muted_still = swap_silence.mp4 (the same still, silent track,
exactly as the open-model runners fed it); sound_on_gray =
series2_gray_<sound>.mp4 built beside this script with the
Series Two Step 2 recipe (plain gray 0x808080 frame, the swap
file's own soundtrack, 10 s).
"""
import base64, csv, json, os, sys, time
import urllib.request, urllib.error

sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
from dynamic_credentials import add_surrogate_to_request  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SWAP = os.path.join(HERE, "..", "..", "pilot2", "swap")
RAW = os.path.join(HERE, "raw")
os.makedirs(RAW, exist_ok=True)

CONNECTOR_NAME = "custom.twelvelabs-run"
ALLOWED_HOSTS = ["api.twelvelabs.io"]
API_URL = "https://api.twelvelabs.io/v1.3/analyze"
MODEL = "pegasus1.6"
REPEATS = 2
DELAY = 3.0

# (sound, version, target) exactly as Test 3 in the open-model CSVs
CELLS = [
    ("silence", "sound_on_still", "silence"),
    ("silence", "muted_still", "silence"),
    ("silence", "sound_on_gray", "silence"),
    ("speech", "sound_on_still", "a person speaking"),
    ("speech", "sound_on_still", "running water"),
    ("speech", "muted_still", "a person speaking"),
    ("speech", "muted_still", "running water"),
    ("speech", "sound_on_gray", "a person speaking"),
    ("speech", "sound_on_gray", "running water"),
    ("beeps", "sound_on_still", "beeping"),
    ("beeps", "muted_still", "beeping"),
    ("beeps", "sound_on_gray", "beeping"),
    ("water", "sound_on_still", "running water"),
    ("water", "sound_on_still", "a person speaking"),
    ("water", "muted_still", "running water"),
    ("water", "muted_still", "a person speaking"),
    ("water", "sound_on_gray", "running water"),
    ("water", "sound_on_gray", "a person speaking"),
    ("vacuum", "sound_on_still", "a vacuum cleaner"),
    ("vacuum", "muted_still", "a vacuum cleaner"),
    ("vacuum", "sound_on_gray", "a vacuum cleaner"),
    ("sizzle", "sound_on_still", "sizzling"),
    ("sizzle", "muted_still", "sizzling"),
    ("sizzle", "sound_on_gray", "sizzling"),
    ("mixed", "sound_on_still", "a person speaking"),
    ("mixed", "sound_on_still", "a vacuum cleaner"),
    ("mixed", "muted_still", "a person speaking"),
    ("mixed", "muted_still", "a vacuum cleaner"),
    ("mixed", "sound_on_gray", "a person speaking"),
    ("mixed", "sound_on_gray", "a vacuum cleaner"),
]


def file_for(sound, version):
    if version == "muted_still":
        return os.path.join(SWAP, "swap_silence.mp4"), "swap_silence.mp4"
    if version == "sound_on_gray":
        name = f"series2_gray_{sound}.mp4"
        return os.path.join(HERE, name), name
    name = f"swap_{sound}.mp4"
    return os.path.join(SWAP, name), name


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
    add_surrogate_to_request(request, CONNECTOR_NAME,
                             allowed_hosts=ALLOWED_HOSTS)
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
    jobs = []
    for sound, version, target in CELLS:
        path, fname = file_for(sound, version)
        if not os.path.exists(path):
            print(f"MISSING FILE: {path}", file=sys.stderr)
            sys.exit(1)
        prompt = f"Is there the sound of {target} in this clip? Answer yes or no."
        for rep in range(1, REPEATS + 1):
            jobs.append((sound, version, target, fname, path, prompt, rep))
    print(f"{len(jobs)} calls queued")
    done = 0
    for sound, version, target, fname, path, prompt, rep in jobs:
        out = os.path.join(
            RAW, f"{sound}__{version}__{target.replace(' ', '_')}__r{rep}.json")
        if os.path.exists(out):
            try:
                rec = json.load(open(out))
                if rec.get("http_status") == 200 and data_field_of(rec.get("raw_body", "")):
                    done += 1
                    continue
            except Exception:
                pass
        status, body = 0, ""
        for attempt in range(4):
            status, body = one_call(path, prompt)
            if status == 200:
                break
            wait = 20 * (2 ** attempt)
            print(f"  http {status}, backing off {wait}s", flush=True)
            time.sleep(wait)
        rec = {"sound": sound, "version": version, "target": target,
               "file": fname, "prompt": prompt, "repeat": rep,
               "http_status": status, "raw_body": body,
               "answer": data_field_of(body)}
        with open(out, "w") as f:
            json.dump(rec, f, indent=1)
        done += 1
        print(f"[{done}/{len(jobs)}] {sound} {version} '{target}' r{rep}: "
              f"{str(rec['answer'])[:60]!r}", flush=True)
        time.sleep(DELAY)
    rows = []
    for fn in sorted(os.listdir(RAW)):
        if fn.endswith(".json"):
            rows.append(json.load(open(os.path.join(RAW, fn))))
    with open(os.path.join(HERE, "pegasus_yesno_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sound", "version", "file", "target", "prompt",
                    "repeat", "http_status", "answer"])
        for r in rows:
            w.writerow([r["sound"], r["version"], r["file"], r["target"],
                        r["prompt"], r["repeat"], r["http_status"], r["answer"]])
    print(f"DONE: {len(rows)} records, CSV written")


if __name__ == "__main__":
    main()
