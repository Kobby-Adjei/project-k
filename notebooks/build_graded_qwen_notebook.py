# build_graded_qwen_notebook.py
# Builds project_k_qwen_graded_colab.ipynb: the graded probes
# (council order item 3, Claude's Nov 4 audit; Kobby's call
# Nov 4: graded probes next, council post after, with
# everything together).
# Step 5's binary checkers sat at the ceiling: 500 vs 1500 Hz
# and 0.5 vs 2.5 s are huge contrasts, so a perfect score cannot
# show decay. This battery regresses the actual values: 360
# tones on a graded grid (30 pitches, log-spaced 300 to
# 2400 Hz, crossed with 12 onsets, 0.25 to 3.25 s), and a small
# ridge readout per state predicts log2(freq) and onset seconds
# from the same two positions step 5 read (audio positions
# mean-pooled, final position). Error by state is the result:
# flat and small means graded fidelity; rising means decay.
# Splits: by clip (primary) and held-out pitch values
# (interpolation check). Nulls: 200 target permutations at
# focus states 16-25, PCA-64 readout.
# The runner is the property battery's own machinery (proof,
# load, feed, token check, generation discipline, extraction),
# with generate_clips and the probe stage replaced.
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))

_ns = {"__file__": os.path.join(HERE, "build_property_qwen_notebook.py"),
       "__name__": "property_builder"}
exec(open(os.path.join(HERE, "build_property_qwen_notebook.py")).read(), _ns)
PROP_SRC = _ns["RUNNER_SRC"]
PROP_CELLS = _ns["cells"]

GEN_AT = PROP_SRC.index("def generate_clips")
EXT_AT = PROP_SRC.index("# --- Extraction")
PROBE_AT = 41902  # the property probe section header

RUNNER_SRC = PROP_SRC[:GEN_AT]

RUNNER_SRC += r'''def generate_clips():
    # The graded stimulus set. 360 tones on a graded grid: 30
    # pitches, log-spaced from 300 to 2400 Hz (three octaves),
    # crossed with 12 onsets from 0.25 to 3.25 s. Every clip is
    # a fresh draw (pitch jitter x0.99 to 1.01, onset jitter
    # +-0.05 s, level -20 +-1 dBFS, random phase) on the shared
    # background scheme, so every waveform is a unique identity
    # and the exact jittered values are the regression targets,
    # stamped in the manifest as freq_hz and start_s. The split
    # is by clip (240 train / 120 test, seed-fixed permutation);
    # the probe stage also computes a held-out-pitch split from
    # the pitch grid index stamped in pitch_band.
    rng = np.random.default_rng(SEED)
    rows = []
    pitches = np.exp(np.linspace(np.log(300.0), np.log(2400.0), 30))
    onsets = [0.25 + k * (3.0 / 11.0) for k in range(12)]
    combos = [(pi, oi) for pi in range(30) for oi in range(12)]
    order = rng.permutation(len(combos))
    train_positions = set(int(x) for x in order[:240])
    for n, (pi, oi) in enumerate(combos):
        freq = float(pitches[pi] * rng.uniform(0.99, 1.01))
        start_s = float(onsets[oi] + rng.uniform(-0.05, 0.05))
        if start_s < 0.05:
            start_s = 0.05
        level = float(-20.0 + rng.uniform(-1.0, 1.0))
        phase = float(rng.uniform(0.0, 2 * np.pi))
        burst_s = 0.5
        wave, floor_db = background(rng)
        n_burst = int(burst_s * SR)
        t = np.arange(n_burst) / SR
        burst = (db_to_amp(level)
                 * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)
        edge = SR // 200
        ramp = 0.5 - 0.5 * np.cos(
            np.linspace(0, np.pi, edge)).astype(np.float32)
        burst[:edge] *= ramp
        burst[-edge:] *= ramp[::-1]
        s0 = int(start_s * SR)
        wave[s0:s0 + n_burst] += burst
        split = "train" if n in train_positions else "test"
        finish_row(rows, f"grad_tone_{n:03d}.wav", "tone", n, wave,
                   PROMPTS["tone"],
                   {"freq_hz": round(freq, 1),
                    "start_s": round(start_s, 3), "burst_s": burst_s,
                    "level_dbfs": round(level, 2), "gain_db": "",
                    "speed_factor": "",
                    "floor_dbfs": floor_db if floor_db is not None else "",
                    "pitch_band": f"p{pi:02d}",
                    "onset_band": f"o{oi:02d}",
                    "split": split})
    print("-" * 70)
    tone_hashes = [r["wave_sha256"] for r in rows if r["group"] == "tone"]
    assert len(set(tone_hashes)) == len(tone_hashes), \
        "duplicate tone waveform hashes; graded identities must be unique"
    print(f"generated {len(rows)} clips; {len(tone_hashes)} tone "
          f"waveforms, all unique")
    from collections import Counter
    splits = Counter(r["split"] for r in rows)
    print(f"  tone: {splits['train']} train, {splits['test']} test")
    def write_manifest(p):
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in MANIFEST_FIELDS})
    save_both("graded_manifest.csv", write_manifest)
    return rows
'''

RUNNER_SRC += PROP_SRC[EXT_AT:PROBE_AT]

RUNNER_SRC += r'''
# --- The graded probes (regression) ------------------------------------------------
# Step 5's binary checkers sat at the ceiling (500 vs 1500 Hz,
# 0.5 vs 2.5 s), so a perfect score could not show decay. Here
# the readout predicts the actual values: log2 of the jittered
# frequency (error in octaves) and the jittered onset in
# seconds, from the same two positions step 5 read (audio
# positions mean-pooled; the final position), at every state.
# Two readouts: pca64 (64 principal components, ridge alpha
# 1.0) and plain (standardized features, ridge alpha 100).
# Two splits: by clip (the generation stamp) and held-out
# pitch (train on even pitch grid indices, test on odd: an
# interpolation check). Nulls: at focus states 16 to 25, the
# pca64 readout on the by-clip split with the training targets
# permuted 200 times; the null R2 mean and 95th percentile
# ride on the matching rows. Verdicts are read outside; this
# stage prints its curves.
import math
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

print("=" * 70)
print("GRADED PROBES: ridge readouts of pitch and onset")
print("=" * 70)

Y = {
    "pitch_log2": np.array([math.log2(float(r["freq_hz"]))
                            for r in rows]),
    "onset_s": np.array([float(r["start_s"]) for r in rows]),
}
PITCH_IDX = np.array([int(r["pitch_band"][1:]) for r in rows])
TR_CLIP = np.array([i for i, r in enumerate(rows)
                    if r["split"] == "train"])
TE_CLIP = np.array([i for i, r in enumerate(rows)
                    if r["split"] == "test"])
TR_HP = np.array([i for i in range(len(rows))
                  if PITCH_IDX[i] % 2 == 0])
TE_HP = np.array([i for i in range(len(rows))
                  if PITCH_IDX[i] % 2 == 1])
SPLITS = {"by_clip": (TR_CLIP, TE_CLIP),
          "heldout_pitch": (TR_HP, TE_HP)}
print(f"by-clip split: {len(TR_CLIP)} train, {len(TE_CLIP)} test")
print(f"held-out-pitch split: {len(TR_HP)} train, "
      f"{len(TE_HP)} test")

def r2_of(y_true, y_pred):
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

def fit_pca64(Xtr, Xte):
    pca = PCA(n_components=64, random_state=0)
    return pca.fit_transform(Xtr), pca.transform(Xte)

# nulls first (focus states), keyed for the row writer
NULLS = {}
FOCUS_STATES = list(range(16, 26))
for state in FOCUS_STATES:
    for pos_name, F in (("audio", feats), ("final", feats_final)):
        X = F[state].astype(np.float64)
        Ztr, Zte = fit_pca64(X[TR_CLIP], X[TE_CLIP])
        for task, y in Y.items():
            rng = np.random.default_rng(SEED + 1000 * state)
            null_r2 = []
            for _ in range(200):
                m = Ridge(alpha=1.0)
                m.fit(Ztr, rng.permutation(y[TR_CLIP]))
                null_r2.append(r2_of(y[TE_CLIP], m.predict(Zte)))
            NULLS[(state, pos_name, task)] = (
                float(np.mean(null_r2)),
                float(np.percentile(null_r2, 95)))
    print(f"  nulls done for state {state}")

G_FIELDS = ["state", "position", "readout", "task", "split",
            "n_train", "n_test", "r2", "mae",
            "null_r2_mean", "null_r2_p95"]
g_rows = []
for state in range(feats.shape[0]):
    for pos_name, F in (("audio", feats), ("final", feats_final)):
        X = F[state].astype(np.float64)
        for split_name, (tr, te) in SPLITS.items():
            Ztr, Zte = fit_pca64(X[tr], X[te])
            mu = X[tr].mean(axis=0)
            sd = X[tr].std(axis=0)
            sd[sd == 0] = 1.0
            for task, y in Y.items():
                preds = {}
                m = Ridge(alpha=1.0)
                m.fit(Ztr, y[tr])
                preds["pca64"] = m.predict(Zte)
                m2 = Ridge(alpha=100.0)
                m2.fit((X[tr] - mu) / sd, y[tr])
                preds["plain"] = m2.predict((X[te] - mu) / sd)
                for readout, yp in preds.items():
                    nm, np95 = "", ""
                    if (split_name == "by_clip"
                            and readout == "pca64"
                            and (state, pos_name, task) in NULLS):
                        nm, np95 = NULLS[(state, pos_name, task)]
                        nm = f"{nm:.4f}"
                        np95 = f"{np95:.4f}"
                    g_rows.append({
                        "state": state, "position": pos_name,
                        "readout": readout, "task": task,
                        "split": split_name,
                        "n_train": len(tr), "n_test": len(te),
                        "r2": f"{r2_of(y[te], yp):.4f}",
                        "mae": f"{float(np.abs(y[te] - yp).mean()):.4f}",
                        "null_r2_mean": nm, "null_r2_p95": np95})
    if state % 12 == 0 or state == feats.shape[0] - 1:
        print(f"  graded readouts done through state {state}")

def write_graded(p):
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=G_FIELDS)
        w.writeheader()
        w.writerows(g_rows)
save_both("graded_probe_results.csv", write_graded)
print(f"graded rows: {len(g_rows)}")

print("GRADED READOUT (mechanical, by-clip split, pca64 readout):")
for task in ("pitch_log2", "onset_s"):
    for pos_name in ("audio", "final"):
        sel = [r for r in g_rows
               if r["task"] == task and r["position"] == pos_name
               and r["readout"] == "pca64" and r["split"] == "by_clip"]
        mae = {int(r["state"]): float(r["mae"]) for r in sel}
        worst = max(mae, key=mae.get)
        print(f"  {task} at {pos_name}: MAE state 0 = "
              f"{mae[0]:.4f}, state 19 = {mae[19]:.4f}, "
              f"state 36 = {mae[36]:.4f}; worst state {worst} "
              f"at {mae[worst]:.4f}")
        if pos_name == "final":
            r0 = [r for r in sel if int(r["state"]) == 0][0]
            print(f"    (final position at state 0, before any "
                  f"layer runs: R2 = {r0['r2']}; this is the "
                  f"built-in sanity line, it should sit at "
                  f"chance)")
print("battery done: graded stimuli generated, features "
      "extracted, ridge readouts and permutation nulls run.")
print("Outputs: graded_probe_results.csv, graded_manifest.csv, "
      "graded_features.npz, graded_run_log.txt.")
'''

RUNNER_SRC = RUNNER_SRC.replace(
    "# qwen_property_run.py - the tracing round 2 runner for",
    "# qwen_graded_run.py - the graded probes runner for", 1)
for old, new in (("property_features", "graded_features"),
                 ("property_attention.csv", "graded_attention.csv"),
                 ("property_run_log.txt", "graded_run_log.txt"),
                 ("property_clips", "graded_clips")):
    RUNNER_SRC = RUNNER_SRC.replace(old, new)
compile(RUNNER_SRC, "qwen_graded_run.py", "exec")
print("runner compiles, chars:", len(RUNNER_SRC))

# ---------------------------------------------------------------- cells
def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.splitlines(keepends=True)}

def code(text):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}

cells = []
cells.append(md(r"""# Project K, Series Three: the graded probes, Qwen2.5-Omni-3B

**What this is.** Step 5's checkers answered two yes/no
questions about a tone (high or low pitch, early or late start)
and scored perfectly at every layer. The council's audit (Nov 4,
2026) named the weakness: those contrasts are huge, so a perfect
score has no room to show decay. This battery asks the graded
question instead: **how many hertz, and at what second**, read
from the model's inside numbers at every state.

- **The tones.** 360 fresh tones on a graded grid: 30 pitches,
  log-spaced from 300 to 2400 Hz, crossed with 12 start times
  from 0.25 to 3.25 s. Each tone is slightly jittered, and the
  exact jittered pitch and start are the targets.
- **The readout.** At every state, a small ridge regression
  reads two places step 5 read (the sound's slots averaged; the
  final slot) and predicts log-pitch and start time. Error by
  state is the result. Two splits: by clip, and held-out pitch
  values (train on half the pitch values, test on the other
  half). 200 target permutations at states 16 to 25 give the
  null.
- **Predictions, written before the run.** If the graded
  information is as intact as the binary result suggested:
  pitch error stays under about 0.06 octaves and start error
  under about 0.2 s at every state. The final slot at state 0
  must sit at chance (no layer has run yet; it has not seen the
  sound). Decay, if it exists, shows as error rising with
  state, most likely at the final slot first.

**The machinery is the property battery's own** (proof, load,
feed, token check, extraction), with the stimulus set and the
readout stage replaced. **Where it runs.** Free Colab account
(kadjei1). **How.** Run the cells from the top, in order.

**The one rule.** We claim only what we ran.
"""))
cells.append(PROP_CELLS[1])   # Step 1 markdown
cells.append(PROP_CELLS[2])   # Step 1 code
cells.append(PROP_CELLS[3])   # Step 2 markdown (masters)
cells.append(PROP_CELLS[4])   # Step 2 code (masters)
cells.append(md(r"""## Step 3 · Write the runner, then check

The runner is written to a plain file and run under this
session's own Python. Check mode runs the proof, the loading
candidates, the token check, and the feed proof, then stops.
The graded tones are generated in battery mode.
"""))
import re as _re
_write = ''.join(PROP_CELLS[6]["source"])
_write = _write.replace("qwen_property_run.py", "qwen_graded_run.py")
_write = _write.replace("property", "graded")
_write = _re.sub(r"RUNNER_SOURCE = '(?:[^'\\]|\\.)*'",
                 lambda m: "RUNNER_SOURCE = " + repr(RUNNER_SRC),
                 _write, count=1)
cells.append(code(_write))
_check = ''.join(PROP_CELLS[7]["source"]).replace(
    "qwen_property_run.py", "qwen_graded_run.py")
cells.append(code(_check))
cells.append(md(r"""## Step 4 · Run the graded battery

Battery mode generates the 360 graded tones, extracts their
features, and runs the ridge readouts with their permutation
nulls. Everything prints live and is also written to
`graded_run_log.txt` in the Drive folder `tracing-qwen3b`.
Results land in the same folder: `graded_probe_results.csv`
and `graded_manifest.csv`.
"""))
_batt = ''.join(PROP_CELLS[9]["source"]).replace(
    "qwen_property_run.py", "qwen_graded_run.py")
cells.append(code(_batt))
cells.append(md(r"""## Closing · What to send back

From the Drive folder `tracing-qwen3b`, the run's own words:

1. the token check outcome;
2. the GRADED READOUT block (MAE at states 0, 19, 36 per task
   and position, with the worst state);
3. the battery exit code;
4. whether `graded_probe_results.csv` and
   `graded_manifest.csv` exist in the folder.
"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3",
                                  "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(HERE, "project_k_qwen_graded_colab.ipynb")
with open(out, "w") as fh:
    json.dump(nb, fh)
print("wrote", out, "cells:", len(cells))
