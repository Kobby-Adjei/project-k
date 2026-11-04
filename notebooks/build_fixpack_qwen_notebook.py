# build_fixpack_qwen_notebook.py
# Builds project_k_qwen_fixpack_colab.ipynb: the property
# patch, the causal step in the council's order (Nov 4, 2026;
# Kobby's go after the closers).
# Design (council spec: ChatGPT's one-property-at-a-time with
# independent donors and sham controls; Claude's partial-layer
# open-form cells; the closers' readout ruling):
#   - Stimuli: the step-5 property tones (360 factorial clips).
#     Recipients and donors are independent instances, matched
#     on the OTHER property, opposed on the patched one.
#   - Patch site: the audio positions only, one decoder layer
#     at a time, grid {0, 14, 16, 18, 19, 20, 22, 24, 28, 34}
#     (34 is the deepest clean layer; 35 is excluded as the
#     known double-norm artifact). Donor states are captured
#     under the same question as the readout.
#   - Readouts, both kept: the normalized yes share (margin)
#     and the emitted first word, under the ORIGINAL questions
#     (the wording battery showed their margins rank at AUC
#     1.0000 / 0.9988; the reversed pitch wording degrades its
#     margin and is not used). Each arm also reads the OTHER
#     property's margin as a specificity control.
#   - Controls: sham patches (donor carries the SAME property
#     value, different instance; the margin should not move)
#     and post-position patches (states after the audio block,
#     final position excluded, at layers {0, 19, 28}; a strict
#     subset that cannot assemble the hybrid state).
#   - Open form at partial layers (Claude): the pitch arm's
#     low-to-high pairs at layers {14, 19, 24} also get the
#     open question under the patch.
# The runner head is the property battery's own head (proof,
# load, feed, token check), taken whole up to clip generation;
# the property clips already exist in the Drive folder.
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))

_ns = {"__file__": os.path.join(HERE, "build_property_qwen_notebook.py"),
       "__name__": "property_builder"}
exec(open(os.path.join(HERE, "build_property_qwen_notebook.py")).read(), _ns)
PROP_SRC = _ns["RUNNER_SRC"]
PROP_CELLS = _ns["cells"]

_ns2 = {"__file__": os.path.join(HERE, "build_closers_qwen_notebook.py"),
        "__name__": "closers_builder"}
exec(open(os.path.join(HERE, "build_closers_qwen_notebook.py")).read(), _ns2)
CLOSERS_CELLS = _ns2["cells"]

CUT = PROP_SRC.index("# --- Probe clip generation")
head = PROP_SRC[:CUT]
head = head.replace("property_", "fixpack_")
head = head.replace("qwen_property_run.py", "qwen_fixpack_run.py")
# fixpack: rms_db_np is defined in the property runner's probe
# tail, which this runner cuts off, but the head's broken-feed
# guard calls it (Claude's Nov 4 residual). Define it in the
# head, just before feed_inputs, so the guard can never
# NameError.
head = head.replace(
    "def feed_inputs",
    "def rms_db_np(x):\n"
    "    if len(x) == 0:\n"
    "        return -120.0\n"
    "    v = float(np.sqrt(np.mean(np.square("
    "x.astype(np.float64)))))\n"
    "    return 20.0 * math.log10(v) if v > 0 else -120.0\n"
    "\ndef feed_inputs", 1)

RUNNER_SRC = head

RUNNER_SRC += r'''
import csv  # the property head keeps its csv import in the
# probe tail this runner cuts off; the first version of this
# battery died on exactly that (NameError after the preflight
# print, Nov 4, 2026). Imports the appended parts need live
# here, not in the borrowed head.

# --- Scorers and questions -------------------------------------------------------
# The original behavioral questions (index check Part B); the
# wording battery (Nov 4, 2026) showed their margins rank the
# properties at AUC 1.0000 (pitch) and 0.9988 (onset), so they
# are the readouts. The reversed pitch wording degraded its
# own margin (0.6331) and is not used.
YES_TOKEN_IDS = []
for _word in ("Yes", " Yes"):
    _enc = processor.tokenizer.encode(_word,
                                      add_special_tokens=False)
    if _enc:
        YES_TOKEN_IDS.append(int(_enc[0]))
YES_TOKEN_IDS = sorted(set(YES_TOKEN_IDS))
NO_TOKEN_IDS = []
for _word in ("No", " No"):
    _enc = processor.tokenizer.encode(_word,
                                      add_special_tokens=False)
    if _enc:
        NO_TOKEN_IDS.append(int(_enc[0]))
NO_TOKEN_IDS = sorted(set(NO_TOKEN_IDS))
PITCH_Q = ("Is the tone in this clip high-pitched? "
           "Answer yes or no.")
ONSET_Q = ("Does the tone start in the first second of this "
           "clip? Answer yes or no.")
P3 = ("What sounds can you hear in this clip? Describe only "
      "what is audible, without guessing from the pictures. "
      "If the audio is silent, say 'silent'.")
print("scorer tokens: yes", YES_TOKEN_IDS, "no", NO_TOKEN_IDS)

def share_and_word(logits):
    probs = torch.softmax(logits.float(), dim=-1)
    py = sum(float(probs[i].item()) for i in YES_TOKEN_IDS)
    pn = sum(float(probs[i].item()) for i in NO_TOKEN_IDS)
    share = py / (py + pn) if (py + pn) > 0 else 0.0
    first = processor.tokenizer.decode(
        [int(torch.argmax(logits).item())])
    return share, first

def margins_for(path):
    """Both questions' margins and words, unpatched."""
    out = {}
    for key, q in (("pitch", PITCH_Q), ("onset", ONSET_Q)):
        inputs, _ = feed_inputs(path, q, expect_silence=False)
        o = thinker_forward(inputs)
        share, first = share_and_word(o.logits[0, -1])
        out[key] = (share, first)
        del o, inputs
        free_memory()
    return out

def audio_block(inputs):
    ids = inputs["input_ids"][0].tolist()
    pos = [i for i, t in enumerate(ids) if t == AUDIO_TOKEN_ID]
    if not pos:
        print("PATCH HARD STOP: no audio positions in a feed.")
        sys.exit(12)
    if pos != list(range(pos[0], pos[-1] + 1)):
        print("PATCH HARD STOP: audio positions not contiguous.")
        sys.exit(12)
    return pos

def post_positions(inputs):
    pos = audio_block(inputs)
    last = int(inputs["input_ids"].shape[1]) - 1
    return list(range(pos[-1] + 1, last))

def patched_share(recip_path, donor_path, layer, mode,
                  question):
    """One patched forward: the donor's state at decoder
    layer `layer` (tuple index layer+1) replaces the
    recipient's rows at the mode's positions, during the
    prefill pass only. Returns the question's yes share and
    first word from the patched logits."""
    r_in, _ = feed_inputs(recip_path, question,
                          expect_silence=False)
    d_in, _ = feed_inputs(donor_path, question,
                          expect_silence=False)
    if int(r_in["input_ids"].shape[1]) != int(
            d_in["input_ids"].shape[1]):
        print("PATCH HARD STOP: donor and recipient rows "
              "differ in length.")
        sys.exit(12)
    d_out = thinker_forward(d_in)
    d_state = d_out.hidden_states[layer + 1][0].detach().clone()
    if mode == "audio":
        pos = audio_block(r_in)
        # fixpack (the council's residual, Nov 4): the donor's
        # state was indexed by the RECIPIENT's position list;
        # assert the donor's own audio span is the same list
        # before pasting, and stop if it is not.
        if audio_block(d_in) != pos:
            print("PATCH HARD STOP: donor and recipient "
                  "audio positions differ.")
            sys.exit(12)
    else:
        pos = post_positions(r_in)
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
    share, first = share_and_word(o.logits[0, -1])
    del o, r_in, d_in, d_out, d_state
    free_memory()
    return share, first

def patched_openform(recip_path, donor_path, layer):
    """The open question (P3) generated with the audio-position
    patch held through the prefill pass."""
    r_in, _ = feed_inputs(recip_path, P3, expect_silence=False)
    d_in, _ = feed_inputs(donor_path, P3, expect_silence=False)
    d_out = thinker_forward(d_in)
    d_state = d_out.hidden_states[layer + 1][0].detach().clone()
    pos = audio_block(r_in)
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
        # The surgery's proven recipe, copied exactly (the
        # handover rerun's generate_from_inputs): the FULL
        # model's generate, no_grad, dict output, batch_decode.
        # This function's first version called
        # THINKER.generate with a bare tensor decode and
        # failed on all 18 cells (Nov 4, 2026); the thinker
        # submodule is not the generation entry point the
        # surgery used.
        with torch.no_grad():
            out = model.generate(
                **r_in, thinker_max_new_tokens=40,
                do_sample=False, return_dict_in_generate=True,
                output_scores=True)
    finally:
        handle.remove()
    new_tokens = out.sequences[:, prefill_len:]
    text = processor.batch_decode(
        new_tokens, skip_special_tokens=True,
        clean_up_tokenization_spaces=False)[0].strip()
    del out, new_tokens, r_in, d_in, d_out, d_state
    free_memory()
    return text

# --- Fixpack additions (the council's Nov 4 corrections) -----
# The four wording-battery forms, byte-identical to the
# closers notebook, plus the two originals. Choice forms are
# scored on the share of the FIRST-named option, the closers
# convention.
PITCH_REV_Q = ("Is the tone in this clip low-pitched? "
               "Answer yes or no.")
ONSET_REV_Q = ("Does the tone start after the first second "
               "of this clip? Answer yes or no.")
PITCH_CHOICE_Q = ("Is the tone high-pitched or low-pitched? "
                  "Answer with one word: high or low.")
ONSET_CHOICE_Q = ("Does the tone start early or late in this "
                  "clip? Answer with one word: early or late.")

def _token_ids(words):
    ids = []
    for w in words:
        enc = processor.tokenizer.encode(
            w, add_special_tokens=False)
        if enc:
            ids.append(int(enc[0]))
    return sorted(set(ids))

HIGH_IDS = _token_ids(("high", " high", "High", " High"))
LOW_IDS = _token_ids(("low", " low", "Low", " Low"))
EARLY_IDS = _token_ids(("early", " early", "Early", " Early"))
LATE_IDS = _token_ids(("late", " late", "Late", " Late"))
print("form tokens: high", HIGH_IDS, "low", LOW_IDS,
      "early", EARLY_IDS, "late", LATE_IDS)

def form_share_word(logits, kind):
    """(share, first word) for one form. yesno: the yes share.
    pitch_choice: the high share. onset_choice: the early
    share (share of the first-named option in each case)."""
    if kind == "yesno":
        return share_and_word(logits)
    pos_ids, neg_ids = (
        (HIGH_IDS, LOW_IDS) if kind == "pitch_choice"
        else (EARLY_IDS, LATE_IDS))
    probs = torch.softmax(logits.float(), dim=-1)
    pp = sum(float(probs[i].item()) for i in pos_ids)
    pn = sum(float(probs[i].item()) for i in neg_ids)
    share = pp / (pp + pn) if (pp + pn) > 0 else 0.0
    first = processor.tokenizer.decode(
        [int(torch.argmax(logits).item())])
    return share, first

FORMS = [("pitch", PITCH_Q, "yesno"),
         ("pitch_reversed", PITCH_REV_Q, "yesno"),
         ("pitch_choice", PITCH_CHOICE_Q, "pitch_choice"),
         ("onset", ONSET_Q, "yesno"),
         ("onset_reversed", ONSET_REV_Q, "yesno"),
         ("onset_choice", ONSET_CHOICE_Q, "onset_choice")]

def capture_audio_state(path, layer):
    """One donor forward under PITCH_Q. The audio positions
    precede the question tokens in the sequence, so under
    causal attention their states at any layer cannot depend
    on the question: one capture serves every wording (the
    council's cheap fix, Nov 4)."""
    d_in, _ = feed_inputs(path, PITCH_Q, expect_silence=False)
    d_out = thinker_forward(d_in)
    state = d_out.hidden_states[layer + 1][0].detach().clone()
    pos = audio_block(d_in)
    del d_in, d_out
    free_memory()
    return state, pos

def patched_form_share(recip_path, d_state, d_pos, layer,
                       question, kind):
    """Paste a pre-captured donor audio state into a recipient
    run under an arbitrary wording; score that wording."""
    r_in, _ = feed_inputs(recip_path, question,
                          expect_silence=False)
    if audio_block(r_in) != d_pos:
        print("PATCH HARD STOP: recipient audio positions "
              "differ from the donor capture's.")
        sys.exit(12)
    pos_t = torch.tensor(d_pos, device=d_state.device)
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
    share, first = form_share_word(o.logits[0, -1], kind)
    del o, r_in
    free_memory()
    return share, first

def patched_share_mediation(recip_path, donor_path, layer,
                            question):
    """The mediation cell: the donor's state at the audio
    positions AND the recipient's own state restored at the
    post-block positions, same layer. If the paste effect
    dies here, the post-block positions are the carrier."""
    r_in, _ = feed_inputs(recip_path, question,
                          expect_silence=False)
    d_in, _ = feed_inputs(donor_path, question,
                          expect_silence=False)
    if int(r_in["input_ids"].shape[1]) != int(
            d_in["input_ids"].shape[1]):
        print("PATCH HARD STOP: donor and recipient rows "
              "differ in length.")
        sys.exit(12)
    a_pos = audio_block(r_in)
    if audio_block(d_in) != a_pos:
        print("PATCH HARD STOP: donor and recipient audio "
              "positions differ.")
        sys.exit(12)
    p_pos = post_positions(r_in)
    d_out = thinker_forward(d_in)
    d_state = d_out.hidden_states[layer + 1][0].detach().clone()
    r_out = thinker_forward(r_in)
    r_state = r_out.hidden_states[layer + 1][0].detach().clone()
    a_t = torch.tensor(a_pos, device=d_state.device)
    p_t = torch.tensor(p_pos, device=d_state.device)
    prefill_len = int(r_in["input_ids"].shape[1])
    layer_mod = THINKER.model.layers[layer]

    def hook(module, args, output):
        hs = output[0] if isinstance(output, tuple) else output
        if hs.shape[1] != prefill_len:
            return output
        new = hs.clone()
        new[0, a_t] = d_state[a_t].to(new.dtype)
        new[0, p_t] = r_state[p_t].to(new.dtype)
        if isinstance(output, tuple):
            return (new,) + output[1:]
        return new

    handle = layer_mod.register_forward_hook(hook)
    try:
        o = thinker_forward(r_in)
    finally:
        handle.remove()
    share, first = share_and_word(o.logits[0, -1])
    del o, r_in, d_in, d_out, r_out, d_state, r_state
    free_memory()
    return share, first

'''

RUNNER_SRC += r'''
# --- The fixpack plan --------------------------------------------------------------
# The council's Nov 4 corrections, as one battery:
#   Part A: gap-matched shams (donor matches the recipient on
#     the TARGET property, differs on the OTHER property by a
#     full gap) at every layer, plus the real pairs at layer
#     21 (the one grid layer the first run lacked, needed
#     beside the mediation cells). word_matches_donor now
#     compares to the donor's OWN unpatched word.
#   Part B: one captured donor state read under all six
#     wordings in the same cells.
#   Part C: the mediation cell (donor at the audio positions,
#     recipient's own state restored at the post positions).
#   Part D: open-form controls (recipients and donors
#     unpatched, real pastes at 28/34, gap-matched sham pastes
#     at 14/19/24/28/34).
print("=" * 70)
print("THE FIXPACK (council corrections, Nov 4)")
print("=" * 70)
# Preflight (kept from the property patch: heal from the
# mounted Drive if the Step 2 copies have not landed, and say
# so in the log; stop with the reason printed, exit 13).
import shutil as _shutil
_need_manifest = not os.path.exists(
    "/content/property_manifest.csv")
_need_clips = (not os.path.isdir("/content/property_clips")
               or len([n for n in os.listdir(
                   "/content/property_clips")
                   if n.endswith(".wav")]) < 400)
if _need_manifest or _need_clips:
    print("preflight: property files missing in this "
          "session; copying from the mounted Drive...")
    try:
        if _need_manifest:
            _shutil.copy(
                "/content/drive/MyDrive/tracing-qwen3b/"
                "property_manifest.csv",
                "/content/property_manifest.csv")
        if _need_clips:
            os.makedirs("/content/property_clips",
                        exist_ok=True)
            _src = ("/content/drive/MyDrive/tracing-qwen3b/"
                    "property_clips")
            for _n in os.listdir(_src):
                if _n.endswith(".wav") and not os.path.exists(
                        os.path.join("/content/property_clips",
                                     _n)):
                    _shutil.copy(os.path.join(_src, _n),
                                 "/content/property_clips")
        print("preflight: healed from Drive.")
    except Exception as _e:
        print("!" * 70)
        print("PREFLIGHT STOP: the property files are not in "
              "place and the Drive copy failed:")
        print(repr(_e))
        print("Step 2 of the notebook did not complete in "
              "this session. Show this log to Muse.")
        print("!" * 70)
        sys.exit(13)
_nwavs = len([n for n in os.listdir("/content/property_clips")
              if n.endswith(".wav")])
print(f"preflight: manifest present, {_nwavs} property "
      f"clips in place")
with open("/content/property_manifest.csv", newline="") as fh:
    manifest = [r for r in csv.DictReader(fh)
                if r["group"] == "tone"]
cells_map = {}
for r in manifest:
    cells_map.setdefault((r["pitch_band"], r["onset_band"]),
                         []).append(r["clip_name"])
for k in cells_map:
    cells_map[k] = sorted(cells_map[k])
print("cells:", {k: len(v) for k, v in cells_map.items()})
truth = {r["clip_name"]: (r["pitch_band"], r["onset_band"])
         for r in manifest}

def clip_path(name):
    return os.path.join("/content/property_clips",
                        name + ".wav")

LAYERS = [0, 14, 16, 18, 19, 20, 21, 22, 24, 28, 34]
WORD_LAYERS = [0, 19, 22]
MED_LAYERS = [19, 20, 21, 22]
OPEN_LAYERS_ALL = [14, 19, 24, 28, 34]

def add_pairs(plan, arm, direction, recip_cell, donor_cell,
              n=3, start=0):
    recips = cells_map[recip_cell][start:start + n]
    donors = cells_map[donor_cell]
    for i, recip in enumerate(recips):
        donor = donors[(i + 1) % len(donors)]
        if donor == recip:
            donor = donors[(i + 2) % len(donors)]
        plan.append((arm, direction, recip, donor))

# The real directional pairs: identical construction to the
# first property-patch run (sorted cells, offset one, first
# three), so this battery's rows join that file's rows.
real_plan = []
for onset in ("early", "late"):
    add_pairs(real_plan, "pitch", "low_to_high",
              ("low", onset), ("high", onset))
    add_pairs(real_plan, "pitch", "high_to_low",
              ("high", onset), ("low", onset))
for pitch in ("low", "high"):
    add_pairs(real_plan, "onset", "early_to_late",
              (pitch, "early"), (pitch, "late"))
    add_pairs(real_plan, "onset", "late_to_early",
              (pitch, "late"), (pitch, "early"))

# Gap-matched shams (the council's fix): the donor matches
# the recipient on the TARGET property and differs on the
# OTHER property by a full gap. Recipients start at index 3
# so no sham pair repeats a real pair or the other sham arm.
sham_plan = []
for onset in ("early", "late"):
    other = "late" if onset == "early" else "early"
    add_pairs(sham_plan, "pitch", "sham_pitch",
              ("low", onset), ("low", other), start=3)
for pitch in ("low", "high"):
    other = "high" if pitch == "low" else "low"
    add_pairs(sham_plan, "onset", "sham_onset",
              (pitch, "early"), (other, "early"), start=3)

# The directional pairs for Parts B and C (a subset of the
# real pairs: one band per arm).
dir_plan = []
add_pairs(dir_plan, "pitch", "low_to_high",
          ("low", "early"), ("high", "early"))
add_pairs(dir_plan, "pitch", "high_to_low",
          ("high", "early"), ("low", "early"))
add_pairs(dir_plan, "onset", "early_to_late",
          ("low", "early"), ("low", "late"))
add_pairs(dir_plan, "onset", "late_to_early",
          ("low", "late"), ("low", "early"))

open_plan = [p for p in real_plan
             if p[0] == "pitch" and p[1] == "low_to_high"]
print(f"plans: real {len(real_plan)}, sham {len(sham_plan)}, "
      f"directional {len(dir_plan)}, open {len(open_plan)}")

names = sorted({p[2] for p in real_plan + sham_plan + dir_plan}
               | {p[3] for p in real_plan + sham_plan + dir_plan})
baselines = {}
for name in names:
    baselines[name] = margins_for(clip_path(name))
print(f"baselines read for {len(baselines)} clips")

PP_FIELDS = ["arm", "direction", "mode", "layer", "recipient",
             "donor", "question", "base_share",
             "donor_base_share", "share_after", "first_word",
             "donor_word", "word_matches_donor"]
pp_rows = []

def _norm_word(w):
    return w.strip().lower().strip(".")

def run_cell(arm, direction, mode, layer, recip, donor):
    for qkey, question in (("pitch", PITCH_Q),
                           ("onset", ONSET_Q)):
        if mode == "mediation":
            share, first = patched_share_mediation(
                clip_path(recip), clip_path(donor), layer,
                question)
        else:
            share, first = patched_share(
                clip_path(recip), clip_path(donor), layer,
                mode, question)
        donor_word = baselines[donor][qkey][1]
        pp_rows.append({
            "arm": arm, "direction": direction, "mode": mode,
            "layer": layer, "recipient": recip, "donor": donor,
            "question": qkey,
            "base_share": f"{baselines[recip][qkey][0]:.4f}",
            "donor_base_share":
                f"{baselines[donor][qkey][0]:.4f}",
            "share_after": f"{share:.4f}",
            "first_word": first,
            "donor_word": donor_word,
            "word_matches_donor":
                _norm_word(first) == _norm_word(donor_word)})

# Part A: gap-matched shams at every layer; real pairs at 21.
print("PART A: gap-matched shams + real pairs at layer 21")
for arm, direction, recip, donor in sham_plan:
    for layer in LAYERS:
        run_cell(arm, direction, "audio", layer, recip, donor)
    print(f"  done {arm}/{direction}: {recip} <- {donor}")
for arm, direction, recip, donor in real_plan:
    run_cell(arm, direction, "audio", 21, recip, donor)
print(f"  done real pairs at layer 21 ({len(real_plan)})")

# Part C: the mediation cell on the directional pairs.
print("PART C: mediation cells")
for arm, direction, recip, donor in dir_plan:
    for layer in MED_LAYERS:
        run_cell(arm, direction, "mediation", layer, recip,
                 donor)
    print(f"  done mediation {arm}/{direction}: "
          f"{recip} <- {donor}")

def write_pp(p):
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=PP_FIELDS)
        w.writeheader()
        w.writerows(pp_rows)
save_both("fixpack_results.csv", write_pp)
print(f"fixpack_results.csv saved: {len(pp_rows)} rows")

# Part B: one captured donor state, all six wordings.
print("PART B: four-wording readout on captured states")
form_base = {}
for name in sorted({p[2] for p in dir_plan}
                   | {p[3] for p in dir_plan}):
    for fname, fq, fkind in FORMS:
        inputs, _ = feed_inputs(clip_path(name), fq,
                                expect_silence=False)
        o = thinker_forward(inputs)
        form_base[(name, fname)] = form_share_word(
            o.logits[0, -1], fkind)
        del o, inputs
        free_memory()
print(f"form baselines read for {len(form_base)} clip-forms")
W_FIELDS = ["arm", "direction", "layer", "recipient", "donor",
            "form", "base_share", "donor_base_share",
            "share_after", "first_word", "donor_word",
            "word_matches_donor"]
w_rows = []
for arm, direction, recip, donor in dir_plan:
    for layer in WORD_LAYERS:
        d_state, d_pos = capture_audio_state(
            clip_path(donor), layer)
        for fname, fq, fkind in FORMS:
            share, first = patched_form_share(
                clip_path(recip), d_state, d_pos, layer, fq,
                fkind)
            donor_word = form_base[(donor, fname)][1]
            w_rows.append({
                "arm": arm, "direction": direction,
                "layer": layer, "recipient": recip,
                "donor": donor, "form": fname,
                "base_share":
                    f"{form_base[(recip, fname)][0]:.4f}",
                "donor_base_share":
                    f"{form_base[(donor, fname)][0]:.4f}",
                "share_after": f"{share:.4f}",
                "first_word": first,
                "donor_word": donor_word,
                "word_matches_donor":
                    _norm_word(first) == _norm_word(donor_word)})
        del d_state
        free_memory()
    print(f"  done wording {arm}/{direction}: "
          f"{recip} <- {donor}")

def write_w(p):
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=W_FIELDS)
        w.writeheader()
        w.writerows(w_rows)
save_both("fixpack_wording.csv", write_w)
print(f"fixpack_wording.csv saved: {len(w_rows)} rows")

# Part D: open-form controls.
print("PART D: open-form controls")

def plain_openform(path):
    """P3 generated with no patch (the surgery recipe)."""
    r_in, _ = feed_inputs(path, P3, expect_silence=False)
    prefill_len = int(r_in["input_ids"].shape[1])
    with torch.no_grad():
        out = model.generate(
            **r_in, thinker_max_new_tokens=40,
            do_sample=False, return_dict_in_generate=True,
            output_scores=True)
    new_tokens = out.sequences[:, prefill_len:]
    text = processor.batch_decode(
        new_tokens, skip_special_tokens=True,
        clean_up_tokenization_spaces=False)[0].strip()
    del out, new_tokens, r_in
    free_memory()
    return text

def _guarded(fn):
    try:
        return fn()
    except Exception as _e:
        import traceback as _tb
        _tb.print_exc(file=sys.stdout)
        return f"ERROR {_e.__class__.__name__}: {_e}"[:200]

OF_FIELDS = ["kind", "recipient", "donor", "layer", "answer"]
of_rows = []
for arm, direction, recip, donor in open_plan:
    onset = truth[recip][1]
    other = "late" if onset == "early" else "early"
    j = cells_map[("low", onset)].index(recip)
    sham_donor = cells_map[("low", other)][
        (j + 1) % len(cells_map[("low", other)])]
    of_rows.append({
        "kind": "unpatched_recipient", "recipient": recip,
        "donor": "", "layer": "",
        "answer": _guarded(lambda: plain_openform(
            clip_path(recip)))})
    of_rows.append({
        "kind": "unpatched_donor", "recipient": recip,
        "donor": donor, "layer": "",
        "answer": _guarded(lambda: plain_openform(
            clip_path(donor)))})
    for layer in (28, 34):
        of_rows.append({
            "kind": "real", "recipient": recip,
            "donor": donor, "layer": layer,
            "answer": _guarded(lambda: patched_openform(
                clip_path(recip), clip_path(donor), layer))})
    for layer in OPEN_LAYERS_ALL:
        of_rows.append({
            "kind": "sham", "recipient": recip,
            "donor": sham_donor, "layer": layer,
            "answer": _guarded(lambda: patched_openform(
                clip_path(recip), clip_path(sham_donor),
                layer))})
    print(f"  done open controls {recip} <- {donor} "
          f"(sham donor {sham_donor})")

def write_of(p):
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=OF_FIELDS)
        w.writeheader()
        w.writerows(of_rows)
save_both("fixpack_openform.csv", write_of)
print(f"fixpack_openform.csv saved: {len(of_rows)} rows")

print("FIXPACK READOUT (mechanical): pct of own gap, "
      "mean per layer")
for arm in ("pitch", "onset"):
    for direction in ("sham_pitch", "sham_onset"):
        rs = [r for r in pp_rows
              if r["arm"] == arm
              and r["direction"] == direction
              and r["mode"] == "audio"]
        if not rs:
            continue
        for qkey in ("pitch", "onset"):
            line = []
            for layer in LAYERS:
                sel = [r for r in rs
                       if int(r["layer"]) == layer
                       and r["question"] == qkey]
                pcts = []
                for r in sel:
                    gap = (float(r["donor_base_share"])
                           - float(r["base_share"]))
                    if abs(gap) > 1e-6:
                        pcts.append(
                            (float(r["share_after"])
                             - float(r["base_share"]))
                            / gap * 100)
                if pcts:
                    line.append(f"L{layer}:"
                                f"{sum(pcts) / len(pcts):.0f}%")
            tag = "own" if qkey == arm else "other"
            print(f"  {direction} read as {qkey} ({tag}): "
                  + " ".join(line))
print("battery done: the fixpack cells ran (gap-matched "
      "shams, real pairs at 21, mediation, four-wording "
      "readout, open-form controls).")
print("Outputs: fixpack_results.csv, fixpack_wording.csv, "
      "fixpack_openform.csv, fixpack_run_log.txt, plus the "
      "head's token check files.")

'''

compile(RUNNER_SRC, "qwen_fixpack_run.py", "exec")
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
cells.append(md(r"""# Project K, Series Three: the fixpack, Qwen2.5-Omni-3B

**What this is.** The corrections battery the council
ordered on Nov 4, after auditing the property patch. The
first run showed a paste at the sound positions transplants
the donor's lean, on a clean depth curve. The audit showed
its sham control could not separate one property from the
whole lean, its word column compared against a label instead
of the donor's own answer, and its open-form cells had no
unpasted controls beside them. This battery rebuilds those
three pieces and adds the two tests the audit asked for
next: one captured donor state read under every wording,
and the mediation cell.

- **Arms.** Pitch (low recipient, high donor, and the reverse)
  and onset (early recipient, late donor, and the reverse).
  Donors are independent clips, matched on the other property.
- **Patch site.** The sound's own positions, one layer at a
  time: 0, 14, 16, 18, 19, 20, 21, 22, 24, 28, 34. (Layer 35 is
  left out on purpose: the index check showed its state is the
  polished one, and pasting it double-polishes.)
- **Readouts.** Both, under the original questions: the yes
  share (the lean) and the first spoken word. Each arm also
  reads the other property's lean, which should NOT move if
  the patch is specific.
- **Controls.** Gap-matched sham pastes: the donor matches
  the recipient on the target property and differs on the
  other property by a full gap, so the sham has something
  real to transplant on the other readout and nothing on
  the target. Every move is reported as a percent of its
  own pair's gap.
- **Open form.** The pitch arm's low-to-high pairs get the
  open question with controls beside it: recipients and
  donors unpatched, real pastes at layers 28 and 34, and
  gap-matched sham pastes at 14, 19, 24, 28, 34.
- **Wording.** Twelve directional pairs are also read with
  one captured donor state under all six wordings (the two
  originals, the two reversed, the two choices).
- **Mediation.** The same pairs get the donor at the sound
  positions with the recipient's own state restored at the
  text positions after the sound, layers 19 to 22.

**Where it runs.** Free Colab account (kadjei1). **How.** Run
the cells from the top, in order. Everything prints live and
is also written to `fixpack_run_log.txt` in the Drive
folder `tracing-qwen3b`.

**The one rule.** We claim only what we ran.
"""))
cells.append(PROP_CELLS[1])   # Step 1 markdown
cells.append(PROP_CELLS[2])   # Step 1 code
cells.append(md(r"""## Step 2 · Get the clips and the masters

The battery needs the step-5 property files (clips plus
`property_manifest.csv`) from the Drive folder
`tracing-qwen3b`, and the master sounds for the head's token
check. The next cell brings the masters; the one after brings
the property files.
"""))
cells.append(PROP_CELLS[4])       # masters for the head
cells.append(CLOSERS_CELLS[5])    # property clips + manifest
cells.append(md(r"""## Step 3 · Write the runner, then check

Check mode runs the proof, the loading candidates, and the
token check, then stops.
"""))
_write = ''.join(PROP_CELLS[6]["source"])
_write = _write.replace("qwen_property_run.py",
                        "qwen_fixpack_run.py")
_write = re.sub(r"RUNNER_SOURCE = '(?:[^'\\]|\\.)*'",
                lambda m: "RUNNER_SOURCE = " + repr(RUNNER_SRC),
                _write, count=1)
cells.append(code(_write))
_check = ''.join(PROP_CELLS[7]["source"]).replace(
    "qwen_property_run.py", "qwen_fixpack_run.py")
cells.append(code(_check))
cells.append(md(r"""## Step 4 · Run the battery

Battery mode reads baselines for every clip in the pair
plans, then runs four parts: gap-matched sham pastes (plus
the real pairs at layer 21), the mediation cells, the
four-wording readout, and the open-form controls. A full
run is long (roughly half an hour or more). Results land
in the Drive folder `tracing-qwen3b`:
`fixpack_results.csv`, `fixpack_wording.csv`,
`fixpack_openform.csv`, `fixpack_run_log.txt`.
"""))
_batt = ''.join(PROP_CELLS[9]["source"]).replace(
    "qwen_property_run.py", "qwen_fixpack_run.py")
cells.append(code(_batt))
cells.append(md(r"""## Closing · What to send back

From the Drive folder `tracing-qwen3b`, the run's own words:

1. the FIXPACK READOUT block;
2. the battery exit code;
3. whether `fixpack_results.csv`, `fixpack_wording.csv`,
   and `fixpack_openform.csv` exist in the folder.
"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3",
                                  "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(HERE, "project_k_qwen_fixpack_colab.ipynb")
with open(out, "w") as fh:
    json.dump(nb, fh)
print("wrote", out, "cells:", len(cells))
