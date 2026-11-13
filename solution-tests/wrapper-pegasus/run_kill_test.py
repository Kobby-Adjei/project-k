#!/usr/bin/env python3
"""Kill test: wrapper fix (Path A) on Pegasus 1.6. Design in DESIGN-FROZEN.md.
Resume-safe: finished jobs with a 200 answer are skipped. Stops on HTTP 422.
W3 depends on that clip's W1 answer, so jobs run in phases.
"""
import base64, csv, json, os, sys, time
import urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
from dynamic_credentials import add_surrogate_to_request  # noqa: E402

RAW = os.path.join(HERE, "raw")
os.makedirs(RAW, exist_ok=True)
OUT = os.path.join(HERE, "kill_test_results.csv")
API_URL = "https://api.twelvelabs.io/v1.3/analyze"
MODEL = "pegasus1.6"
DELAY = 3.0
CONNECTOR = os.environ.get("TL_CONNECTOR", "custom.twelvelabs-temp4")

P2 = "Describe what happens in this video, including any sounds that are audible."
P3 = ("What sounds can you hear in this clip? Describe only what is audible, "
      "without guessing from the pictures. If the audio is silent, say 'silent'.")
R2 = "What happens in this video? Mention any sounds you can hear."
R3 = "Summarize this video and list any audible sounds."

CLIPS = {
    "water": (os.path.join(ROOT, "pilot2", "swap", "swap_water.mp4"), "running water"),
    "vacuum": (os.path.join(ROOT, "pilot2", "swap", "swap_vacuum.mp4"), "a vacuum cleaner"),
    "applause": (os.path.join(ROOT, "docs-sounds-check", "docs_applause_still.mp4"), "applause"),
    "silence": (os.path.join(ROOT, "pilot2", "swap", "swap_silence.mp4"), "running water"),
}
LURE = {"water": "a vacuum cleaner", "vacuum": "applause", "applause": "running water"}


def yn(target):
    return f"Is there the sound of {target} in this clip? Answer yes or no."


def w3_prompt(w1_answer):
    return (f"A sound-only check of this same clip reported: '{w1_answer}' "
            "Now describe what happens in this video, including any sounds "
            "that are audible. Include the sounds from that sound-only report "
            "in your answer.")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args):
        raise RuntimeError("redirect refused, verify the endpoint")


def one_call(path, prompt):
    with open(path, "rb") as fh:
        file_bytes = fh.read()
    payload = {"model_name": MODEL,
               "video": {"type": "base64_string",
                         "base64_string": base64.b64encode(file_bytes).decode("ascii")},
               "prompt": prompt, "temperature": 0, "stream": False}
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(API_URL, data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    add_surrogate_to_request(request, CONNECTOR, allowed_hosts=["api.twelvelabs.io"])
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
    return obj.get("data") if isinstance(obj, dict) else None


def run_job(sound, arm, target, prompt):
    path, _ = CLIPS[sound]
    tag = f"{sound}__{arm}"
    out = os.path.join(RAW, f"{tag}.json")
    if os.path.exists(out):
        try:
            rec = json.load(open(out))
            if rec.get("http_status") == 200 and rec.get("answer"):
                print(f"skip {tag}: already saved", flush=True)
                return rec
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
        print(f"{tag}: HTTP 422 - allowance gone, STOP", flush=True)
        print(body[:300], flush=True)
        sys.exit(22)
    rec = {"sound": sound, "arm": arm, "target": target, "prompt": prompt,
           "file": os.path.basename(path), "http_status": status,
           "raw_body": body, "answer": data_field_of(body)}
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"{tag}: HTTP {status} answer={str(rec['answer'])[:80]!r}", flush=True)
    time.sleep(DELAY)
    return rec


def main():
    for sound, (path, target) in CLIPS.items():
        if not os.path.exists(path):
            print(f"MISSING FILE: {path}", file=sys.stderr)
            return 1
    # Phase 1: everything except W3
    w1_answers = {}
    for sound in ("water", "vacuum", "applause"):
        _, target = CLIPS[sound]
        run_job(sound, "S", target, P2)
        run_job(sound, "R1", target, P2)
        run_job(sound, "R2", target, R2)
        run_job(sound, "R3", target, R3)
        rec = run_job(sound, "W1", target, P3)
        w1_answers[sound] = rec["answer"]
        run_job(sound, "W2", target, yn(target))
        run_job(sound, "LURE", LURE[sound], yn(LURE[sound]))
    _, s_target = CLIPS["silence"]
    run_job("silence", "S", s_target, P2)
    rec = run_job("silence", "W1", s_target, P3)
    w1_answers["silence"] = rec["answer"]
    run_job("silence", "W2", s_target, yn(s_target))
    # Phase 2: W3 carry-forward, built from each clip's own W1 answer
    for sound in ("water", "vacuum", "applause", "silence"):
        _, target = CLIPS[sound]
        run_job(sound, "W3", target, w3_prompt(w1_answers[sound]))
    rows = []
    for fn in sorted(os.listdir(RAW)):
        if fn.endswith(".json"):
            rows.append(json.load(open(os.path.join(RAW, fn))))
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sound", "arm", "target", "prompt", "file", "http_status", "answer"])
        for r in rows:
            w.writerow([r["sound"], r["arm"], r["target"], r["prompt"],
                        r["file"], r["http_status"], r["answer"]])
    print(f"DONE: {len(rows)} records, CSV written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
