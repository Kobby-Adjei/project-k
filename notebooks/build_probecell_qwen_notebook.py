# build_probecell_qwen_notebook.py
# Builds project_k_qwen_probecell_colab.ipynb: the probe cell
# (Claude's separating cell from the Nov 4 fixpack audits,
# Kobby's go Nov 4, 2026: "Run the probe cell, then the
# corrected relay test").
#
# The question: the yes/no lean carries no within-band pitch
# signal (regression on the behavioral file: corr 0.001 /
# 0.014 within band), and a within-band pitch paste does not
# move the lean in the mid-window. Is the fine pitch still
# IN the pasted state, readable by the graded probe
# downstream of the paste (the lean is a coarse readout of a
# fine state), or does the pasted state itself go band-coarse
# mid-window (the state is coarse, the lean is innocent)?
#
# Design:
#   - Probe: ridge regression of log2(freq) on PCA-64 of the
#     hidden states (audio-mean and final positions), the
#     graded battery's pca64 readout, trained in-run on 96
#     fresh tones (24 log-spaced freqs 300-2400 Hz x 4
#     onsets), states extracted under the pitch question.
#   - Pairs: wide within-band pitch pairs, onset matched:
#     (400, 900) on the low side, (1600, 2200) on the high
#     side, both directions, 2 instances each = 8 pairs.
#     Their lean gap is expected small (that is the premise);
#     baselines are read and the gap is printed in steps.
#   - Paste layers {14, 16, 18, 20} at the audio positions.
#     The patched run's states are probed at the paste site
#     (sanity: must read donor) and at every downstream state,
#     audio-mean and final positions. The same run's logits
#     give the lean (premise check: stays at the recipient).
#   - Controls: the probe reads all 16 pair clips unpatched
#     (in-run validity: MAE should land near the graded
#     battery's 0.05-0.09 octaves).
# Registered predictions:
#   1. Unpatched probe MAE <= ~0.10 octaves at audio
#      positions, states 5-36.
#   2. Paste site reads the donor (error < 0.1 octaves).
#   3. The cell: downstream of the paste, predictions sit
#      closer to the DONOR's frequency for most pairs and
#      layers (state carries fine pitch; the lean ignores
#      it), OR stay at the recipient's (the state itself is
#      band-coarse mid-window). Either outcome is a result.
# Runner: the fixpack runner's head + machinery whole (donor
# span check, own imports, preflight discipline), battery
# replaced. Stimuli are generated in-notebook on digital
# silence (no background scheme; the probe trains and is
# tested on the same recipe, so the cell is internally
# valid; absolute MAE is compared to the graded battery
# with that caveat stated).
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))

_ns = {"__file__": os.path.join(HERE, "build_fixpack_qwen_notebook.py"),
       "__name__": "fixpack_builder"}
exec(open(os.path.join(HERE, "build_fixpack_qwen_notebook.py")).read(), _ns)
FIX_SRC = _ns["RUNNER_SRC"]
PROP_CELLS = _ns["PROP_CELLS"]
CLOSERS_CELLS = _ns["CLOSERS_CELLS"]
code = _ns["code"]
md = _ns["md"]

CUT = FIX_SRC.index("# --- The fixpack plan")
RUNNER_SRC = FIX_SRC[:CUT]
RUNNER_SRC = RUNNER_SRC.replace("fixpack_", "probecell_")
RUNNER_SRC = RUNNER_SRC.replace("qwen_fixpack_run.py",
                                 "qwen_probecell_run.py")

RUNNER_SRC += r'''
# --- The probe cell --------------------------------------------------------------
import csv
import numpy as np
import soundfile as sf
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

print("=" * 70)
print("THE PROBE CELL (graded probe read on patched states)")
print("=" * 70)

SR = 16000
CLIP_DIR = "/content/probecell_clips"
os.makedirs(CLIP_DIR, exist_ok=True)

def synth_tone(freq, start_s, level_db, phase, seed_tag):
    """One 4.0 s clip: a 0.5 s sine burst on digital silence,
    the property recipe's ramps (edge = SR//200)."""
    wave = np.zeros(SR * 4, dtype=np.float32)
    n_burst = int(0.5 * SR)
    t = np.arange(n_burst) / SR
    amp = float(10 ** (level_db / 20.0))
    burst = (amp * np.sin(2 * np.pi * freq * t + phase)
             ).astype(np.float32)
    edge = SR // 200
    ramp = 0.5 - 0.5 * np.cos(
        np.linspace(0, np.pi, edge)).astype(np.float32)
    burst[:edge] *= ramp
    burst[-edge:] *= ramp[::-1]
    s0 = int(start_s * SR)
    wave[s0:s0 + n_burst] += burst
    return wave

def write_clip(name, wave):
    path = os.path.join(CLIP_DIR, name)
    sf.write(path, wave, SR, subtype="PCM_16")
    back, _ = sf.read(path, dtype="float32")
    if len(back) != len(wave) or float(
            np.sqrt(np.mean(back ** 2))) <= 0:
        print("PROBE HARD STOP: clip read-back failed:", name)
        sys.exit(7)
    return path

rng = np.random.default_rng(20261106)

# Training tones: 24 log-spaced freqs x 4 onsets = 96.
train = []
freqs = np.exp(np.linspace(np.log(300.0), np.log(2400.0), 24))
for fi, f in enumerate(freqs):
    for oi, onset in enumerate((0.5, 1.25, 2.0, 2.75)):
        lvl = float(-20.0 + rng.uniform(-1.0, 1.0))
        ph = float(rng.uniform(0.0, 2 * np.pi))
        w = synth_tone(float(f), onset, lvl, ph, fi)
        p = write_clip(f"train_{fi:02d}_{oi}.wav", w)
        train.append((p, float(f)))
print("training clips:", len(train))

# Pairs: wide within-band, onset matched at 1.5 s.
pair_specs = []
for lo, hi in ((400.0, 900.0), (1600.0, 2200.0)):
    for a, b in ((lo, hi), (hi, lo)):
        for inst in range(2):
            pair_specs.append((a, b, inst))
pairs = []
for pi, (rf, df, inst) in enumerate(pair_specs):
    out = {}
    for tag, f in (("recip", rf), ("donor", df)):
        jit = float(rng.uniform(0.99, 1.01))
        lvl = float(-20.0 + rng.uniform(-1.0, 1.0))
        ph = float(rng.uniform(0.0, 2 * np.pi))
        w = synth_tone(f * jit, 1.5, lvl, ph, pi)
        out[tag] = (write_clip(f"pair{pi}_{tag}.wav", w),
                    f * jit)
    pairs.append({"id": f"pair{pi}", "recip": out["recip"],
                  "donor": out["donor"]})
print("pairs:", len(pairs))

def state_features(path):
    """(37, 2, H) float32: per state, [audio-mean, final]."""
    inputs, _ = feed_inputs(path, PITCH_Q, expect_silence=False)
    pos = audio_block(inputs)
    out = thinker_forward(inputs)
    hs = out.hidden_states
    feats = np.empty((len(hs), 2, hs[0].shape[-1]),
                     dtype=np.float32)
    for s, h in enumerate(hs):
        hv = h[0].float()
        feats[s, 0] = hv[pos].mean(0).cpu().numpy()
        feats[s, 1] = hv[-1].cpu().numpy()
    del out, inputs
    free_memory()
    return feats

print("extracting training states...")
X = np.stack([state_features(p) for p, _ in train])
y = np.log2(np.array([f for _, f in train]))
print("training features:", X.shape)

probes = {}
for s in range(X.shape[1]):
    for pi_, pname in ((0, "audio"), (1, "final")):
        Xtr = X[:, s, pi_, :]
        pca = PCA(n_components=64, random_state=0)
        Z = pca.fit_transform(Xtr)
        m = Ridge(alpha=1.0)
        m.fit(Z, y)
        probes[(s, pname)] = (pca, m)
print("probes fitted:", len(probes))

def probe_pred(feats, s, pname):
    pca, m = probes[(s, pname)]
    z = pca.transform(feats[s:s + 1, {"audio": 0, "final": 1}[pname]])
    return float(2 ** float(m.predict(z)[0]))

# Controls: probe every pair clip unpatched.
probe_rows = []
lean_rows = []
baselines = {}
for pr in pairs:
    for tag in ("recip", "donor"):
        path, f = pr[tag]
        feats = state_features(path)
        baselines[(pr["id"], tag)] = margins_for(path)
        for s in range(37):
            for pname in ("audio", "final"):
                pred = probe_pred(feats, s, pname)
                probe_rows.append({
                    "kind": "unpatched_" + tag,
                    "pair": pr["id"], "layer_pasted": "",
                    "state_read": s, "position": pname,
                    "pred_hz": round(pred, 1),
                    "recip_hz": round(pr["recip"][1], 1),
                    "donor_hz": round(pr["donor"][1], 1),
                    "err_vs_recip_oct": round(abs(
                        np.log2(pred / pr["recip"][1])), 4),
                    "err_vs_donor_oct": round(abs(
                        np.log2(pred / pr["donor"][1])), 4)})
        del feats
print("unpatched controls probed")

def patched_forward_states(recip_path, donor_path, layer):
    """The fixpack paste (audio positions, donor span
    checked), returning the patched run's hidden states as
    (37, 2, H) features plus its pitch-question logits."""
    r_in, _ = feed_inputs(recip_path, PITCH_Q,
                          expect_silence=False)
    d_in, _ = feed_inputs(donor_path, PITCH_Q,
                          expect_silence=False)
    if int(r_in["input_ids"].shape[1]) != int(
            d_in["input_ids"].shape[1]):
        print("PATCH HARD STOP: length mismatch.")
        sys.exit(12)
    pos = audio_block(r_in)
    if audio_block(d_in) != pos:
        print("PATCH HARD STOP: donor audio positions "
              "differ.")
        sys.exit(12)
    d_out = thinker_forward(d_in)
    d_state = d_out.hidden_states[layer + 1][0].detach().clone()
    pos_t = torch.tensor(pos, device=d_state.device)
    prefill_len = int(r_in["input_ids"].shape[1])
    layer_mod = THINKER.model.layers[layer]

    def hook(module, args, output):
        hs = output[0] if isinstance(output, tuple) else output
        if hs.shape[1] != prefill_len:
            return output
        new = hs.clone()
        new[0, pos_t] = d_state[pos_t].to(new.dtype)
        if isinstance(output, tuple):
            return (new,) + output[1:]
        return new

    handle = layer_mod.register_forward_hook(hook)
    try:
        o = thinker_forward(r_in)
    finally:
        handle.remove()
    hs = o.hidden_states
    feats = np.empty((len(hs), 2, hs[0].shape[-1]),
                     dtype=np.float32)
    for s, h in enumerate(hs):
        hv = h[0].float()
        feats[s, 0] = hv[pos].mean(0).cpu().numpy()
        feats[s, 1] = hv[-1].cpu().numpy()
    share, first = share_and_word(o.logits[0, -1])
    del o, r_in, d_in, d_out, d_state
    free_memory()
    return feats, share, first

PASTE_LAYERS = (14, 16, 18, 20)
for pr in pairs:
    rp, rf = pr["recip"]
    dp, df = pr["donor"]
    rb = baselines[(pr["id"], "recip")]["pitch"][0]
    db = baselines[(pr["id"], "donor")]["pitch"][0]
    for layer in PASTE_LAYERS:
        try:
            feats, share, first = patched_forward_states(
                rp, dp, layer)
        except Exception:
            import traceback
            traceback.print_exc(file=sys.stdout)
            raise
        lean_rows.append({
            "pair": pr["id"], "layer": layer,
            "recip_hz": round(rf, 1), "donor_hz": round(df, 1),
            "base_share_recip": round(rb, 4),
            "base_share_donor": round(db, 4),
            "patched_share": round(share, 4),
            "first_word": first})
        for s in range(layer + 1, 37):
            for pname in ("audio", "final"):
                pred = probe_pred(feats, s, pname)
                er = abs(np.log2(pred / rf))
                ed = abs(np.log2(pred / df))
                probe_rows.append({
                    "kind": "patched",
                    "pair": pr["id"], "layer_pasted": layer,
                    "state_read": s, "position": pname,
                    "pred_hz": round(pred, 1),
                    "recip_hz": round(rf, 1),
                    "donor_hz": round(df, 1),
                    "err_vs_recip_oct": round(float(er), 4),
                    "err_vs_donor_oct": round(float(ed), 4)})
        del feats
    print("pair done:", pr["id"], flush=True)

def save_csv(rows, name, fields):
    def _w(p):
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    save_both(name, _w)
    print("saved", name, len(rows), "rows")

save_csv(probe_rows, "probecell_probe.csv",
         ["kind", "pair", "layer_pasted", "state_read",
          "position", "pred_hz", "recip_hz", "donor_hz",
          "err_vs_recip_oct", "err_vs_donor_oct"])
save_csv(lean_rows, "probecell_lean.csv",
         ["pair", "layer", "recip_hz", "donor_hz",
          "base_share_recip", "base_share_donor",
          "patched_share", "first_word"])

print("=" * 70)
print("PROBE CELL READOUT")
print("=" * 70)
import collections
# validity: unpatched audio-position MAE, states 5-36
v = [r["err_vs_recip_oct"] if r["kind"] == "unpatched_recip"
     else r["err_vs_donor_oct"] for r in probe_rows
     if r["kind"].startswith("unpatched")
     and r["position"] == "audio" and r["state_read"] >= 5]
print("unpatched validity MAE (audio, states 5-36): %.4f oct"
      % (sum(v) / len(v)))
site = [r for r in probe_rows if r["kind"] == "patched"
        and r["state_read"] == r["layer_pasted"] + 1]
print("paste-site rows: %d, mean err vs donor %.4f oct"
      % (len(site), sum(r["err_vs_donor_oct"] for r in site)
         / max(len(site), 1)))
for layer in PASTE_LAYERS:
    for pname in ("audio", "final"):
        d = [r for r in probe_rows if r["kind"] == "patched"
             and r["layer_pasted"] == layer
             and r["position"] == pname
             and r["state_read"] >= layer + 2]
        cd = sum(1 for r in d
                 if r["err_vs_donor_oct"] < r["err_vs_recip_oct"])
        print("layer %d %s: downstream rows %d, closer to "
              "donor %d (%.0f%%), mean err donor %.3f / "
              "recip %.3f oct"
              % (layer, pname, len(d), cd,
                 100.0 * cd / max(len(d), 1),
                 sum(r["err_vs_donor_oct"] for r in d)
                 / max(len(d), 1),
                 sum(r["err_vs_recip_oct"] for r in d)
                 / max(len(d), 1)))
print("PROBE CELL DONE")
'''

cells = []
cells.append(md(r"""# Project K · Qwen 3B · The probe cell (Series Three)

**The question.** The yes/no lean carries no within-band
pitch: asked "high-pitched?", the model's pull separates the
two pitch bands perfectly and ignores the exact pitch inside
a band (checked on the 360 tones: the link inside a band is
0.001 to 0.014). A paste of a same-band donor's states does
not move the lean in the middle layers. So is the fine pitch
still inside the pasted state, readable by a fine checker,
while the lean reads it coarsely? Or is the pasted state
itself coarse in those layers?

**What runs.** A fine pitch checker (the graded battery's
readout: ridge regression on PCA-64 of the hidden states) is
trained in-run on 96 fresh tones. Then, for 8 wide
within-band pairs (400 vs 900 Hz, 1600 vs 2200 Hz, both
directions), the donor's states are pasted at layers 14, 16,
18, 20, and the checker reads the patched run's states at the
paste site and every layer after it, at the sound's positions
and at the final answer position. The same runs report the
lean, which should stay at the recipient's.

**Where it runs.** Free Colab account (kadjei1). **How.** Run
the cells from the top, in order. Everything prints live and
is also written to `probecell_run_log.txt` in the Drive
folder `tracing-qwen3b`.

**The one rule.** We claim only what we ran.
"""))
cells.append(PROP_CELLS[1])   # Step 1 markdown
cells.append(PROP_CELLS[2])   # Step 1 code
cells.append(md(r"""## Step 2 · Get the masters

The head's token check needs the master sounds from the
Drive folder `tracing-qwen3b`. The probe cell's own clips are
generated inside the runner; no other files are needed.
"""))
cells.append(PROP_CELLS[4])       # masters for the head
cells.append(md(r"""## Step 3 · Write the runner, then check

Check mode runs the proof, the loading candidates, and the
token check, then stops.
"""))
_write = ''.join(PROP_CELLS[6]["source"])
_write = _write.replace("qwen_property_run.py",
                        "qwen_probecell_run.py")
_write = re.sub(r"RUNNER_SOURCE = '(?:[^'\\]|\\.)*'",
                lambda m: "RUNNER_SOURCE = " + repr(RUNNER_SRC),
                _write, count=1)
cells.append(code(_write))
_check = ''.join(PROP_CELLS[7]["source"]).replace(
    "qwen_property_run.py", "qwen_probecell_run.py")
cells.append(code(_check))
cells.append(md(r"""## Step 4 · Run the battery

Battery mode trains the checker, probes the unpatched
controls, then runs the pastes. Results land in the Drive
folder `tracing-qwen3b`: `probecell_probe.csv`,
`probecell_lean.csv`, `probecell_run_log.txt`.
"""))
_batt = ''.join(PROP_CELLS[9]["source"]).replace(
    "qwen_property_run.py", "qwen_probecell_run.py")
cells.append(code(_batt))
cells.append(md(r"""## Closing · What to send back

From the Drive folder `tracing-qwen3b`, the run's own words:

1. the PROBE CELL READOUT block;
2. the battery exit code;
3. whether `probecell_probe.csv` and `probecell_lean.csv`
   exist in the folder.
"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3",
                                  "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(HERE, "project_k_qwen_probecell_colab.ipynb")
with open(out, "w") as fh:
    json.dump(nb, fh)
print("wrote", out, "cells:", len(cells))
