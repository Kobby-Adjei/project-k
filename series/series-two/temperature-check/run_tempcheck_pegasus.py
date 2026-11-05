#!/usr/bin/env python3
"""Pegasus temperature arm (Nov 5, 2026).

The Series One / Pilot Two Pegasus answers were all taken at
temperature 0. This arm re-asks the claim-carrying cells at
temperature 0.7, 3 draws per cell, same files and prompts
byte-identical to the original runs:

- swap files x P3 (plain sounds question, from run_swap.py)
- real clips x P1 (repaired JSON prompt, imported from
  run_eggs.py so it is exact by construction)

30 calls. Resume-safe: raw bodies saved per job under
raw-pegasus/, finished jobs skipped. Stops the whole run on
HTTP 422 (allowance gone again, exit 22). One backoff retry
on HTTP 429. Output: tempcheck_pegasus_results.csv.
"""
import base64, csv, json, os, sys, time
import urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT2 = os.path.join(HERE, "..", "pilot2")
sys.path.insert(0, PILOT2)
sys.path.insert(0, os.path.join(PILOT2, "swap"))
sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
from dynamic_credentials import add_surrogate_to_request  # noqa: E402
import run_eggs  # noqa: E402  (exact P1 prompt)
import run_swap  # noqa: E402  (exact P3 prompt)

RAW = os.path.join(HERE, "raw-pegasus")
os.makedirs(RAW, exist_ok=True)
OUT = os.path.join(HERE, "tempcheck_pegasus_results.csv")
API_URL = "https://api.twelvelabs.io/v1.3/analyze"
MODEL = "pegasus1.6"
TEMPERATURE = 0.7
DRAWS = 3
DELAY = 3.0

P1 = run_eggs.PROMPT
P3 = run_swap.PROMPTS["P3"]

JOBS = []  # (cell, path, prompt_id, prompt)
for sound in ("silence", "speech", "beeps", "water", "vacuum",
              "sizzle", "mixed"):
    JOBS.append((f"swap_{sound}",
                 os.path.join(PILOT2, "swap", f"swap_{sound}.mp4"),
                 "P3", P3))
for cell, path in (
        ("real_vacuum_P02_07_5", os.path.join(PILOT2, "clips", "P02_07_5.mp4")),
        ("real_jar_P32_01_54", os.path.join(PILOT2, "clips", "P32_01_54.mp4")),
        ("real_eggs_own-01", os.path.join(PILOT2, "own-recordings", "own-01.mp4"))):
    JOBS.append((cell, path, "P1", P1))


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
        "temperature": TEMPERATURE,
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(API_URL, data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    add_surrogate_to_request(request,
                             os.environ.get("TL_CONNECTOR",
                                            "custom.twelvelabs-run"),
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
    rows = []
    if os.path.exists(OUT):
        with open(OUT, newline="") as fh:
            rows = list(csv.DictReader(fh))
    done = {(r["cell"], r["prompt_id"], int(r["draw"])) for r in rows}
    total = len(JOBS) * DRAWS
    n = 0
    for cell, path, pid, prompt in JOBS:
        for draw in range(1, DRAWS + 1):
            n += 1
            if (cell, pid, draw) in done:
                print(f"[{n}/{total}] {cell} {pid} draw {draw}: already saved, skip")
                continue
            status, body = one_call(path, prompt)
            if status == 429:
                print(f"[{n}/{total}] {cell} {pid} draw {draw}: HTTP 429, back off 30s, retry once")
                time.sleep(30)
                status, body = one_call(path, prompt)
            raw_path = os.path.join(RAW, f"{cell}__{pid}__d{draw}.json")
            with open(raw_path, "w") as fh:
                fh.write(body)
            if status == 422:
                print(f"[{n}/{total}] {cell} {pid} draw {draw}: HTTP 422 - allowance gone, STOP")
                print(body[:300])
                write_out(rows)
                return 22
            answer = data_field_of(body)
            rows.append({
                "cell": cell, "file": os.path.basename(path),
                "prompt_id": pid, "temperature": TEMPERATURE,
                "draw": draw, "http_status": status,
                "answer": answer if answer is not None else "",
            })
            print(f"[{n}/{total}] {cell} {pid} draw {draw}: HTTP {status} "
                  f"answer={str(answer)[:90]!r}")
            write_out(rows)
            time.sleep(DELAY)
    print(f"DONE: {len(rows)} rows in {OUT}")
    return 0


def write_out(rows):
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["cell", "file", "prompt_id",
                                           "temperature", "draw",
                                           "http_status", "answer"])
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    sys.exit(main())
