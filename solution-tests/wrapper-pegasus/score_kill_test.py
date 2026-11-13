#!/usr/bin/env python3
"""Score the kill test from the raw answer files only. Rules in DESIGN-FROZEN.md."""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")

SYNONYMS = {
    "water": [r"\bwater\b", r"\brunning water\b", r"\btap\b", r"\bfaucet\b", r"\btrickling\b", r"\bdripping\b"],
    "vacuum": [r"\bvacuum\b", r"\bhoover\b"],
    "applause": [r"\bapplause\b", r"\bclapping\b", r"\bclaps\b"],
}
ALL_SOUNDS = ["water", "vacuum", "applause"]


def names(answer, sound):
    if not answer:
        return False
    a = answer.lower()
    # denial phrases override: an answer that calls the clip silent names nothing
    return any(re.search(p, a) for p in SYNONYMS[sound])


def load():
    recs = {}
    for fn in sorted(os.listdir(RAW)):
        if fn.endswith(".json"):
            r = json.load(open(os.path.join(RAW, fn)))
            recs[(r["sound"], r["arm"])] = r
    return recs


def main():
    recs = load()
    print(f"records: {len(recs)} (expected 34)")
    table = {}
    for sound in ("water", "vacuum", "applause", "silence"):
        row = {}
        for arm in ("S", "R1", "R2", "R3", "W1", "W2", "W3", "LURE"):
            r = recs.get((sound, arm))
            if r is None:
                continue
            ans = r["answer"] or ""
            if arm in ("W2", "LURE"):
                row[arm] = ans.strip()
            elif sound == "silence":
                row[arm] = [s for s in ALL_SOUNDS if names(ans, s)] or "none"
            else:
                hit = names(ans, sound)
                wrong = [s for s in ALL_SOUNDS if s != sound and names(ans, s)]
                row[arm] = ("HIT" if hit else "miss") + (f"+wrong:{','.join(wrong)}" if wrong else "")
        table[sound] = row
        print(sound, row)
    # Verdict per frozen bar
    w_hits = sum(1 for s in ("water", "vacuum", "applause")
                 if table[s].get("W3", "").startswith("HIT"))
    s_hits = sum(1 for s in ("water", "vacuum", "applause")
                 if table[s].get("S", "").startswith("HIT"))
    r_hits = sum(1 for s in ("water", "vacuum", "applause")
                 if any(table[s].get(a, "").startswith("HIT") for a in ("R1", "R2", "R3")))
    silence_clean = table["silence"].get("W3") == "none"
    print(f"\nS hits: {s_hits}/3   R (any of 3) hits: {r_hits}/3   W3 hits: {w_hits}/3   silence W3 clean: {silence_clean}")
    if w_hits >= 2 and w_hits > s_hits and w_hits > r_hits and silence_clean:
        print("VERDICT: wrapper SURVIVES the kill test")
    elif w_hits <= 1 or r_hits >= w_hits:
        print("VERDICT: wrapper KILLED as a fix (per frozen bar)")
    else:
        print("VERDICT: mixed, read the table")
    return 0


if __name__ == "__main__":
    sys.exit(main())
