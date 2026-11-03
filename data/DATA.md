# DATA — EPIC-KITCHENS-100 for the Project K pilot (EXPLORATORY)

## Sources (all official, no account, no form, no signature)

- Annotations: https://github.com/epic-kitchens/epic-kitchens-100-annotations
  (cloned to `data/annotations/`). License file in the repo: **CC BY-NC 4.0**
  (`data/annotations/license.txt`). EK-100 itself is released under CC BY-NC 4.0
  for non-commercial research; this pilot is non-commercial evaluation research.
- Download machinery: https://github.com/epic-kitchens/epic-kitchens-download-scripts
  (cloned to `data/download-scripts/`). The downloader fetches files directly —
  **there is NO license click-through, form, or signature anywhere in the official
  path. Kobby does not need to sign anything for the data leg.**
- Videos are hosted at the University of Bristol data.bris.ac.uk under two
  dataset roots, because EK-100 reuses EPIC-KITCHENS-55 files for older
  participants:
  - EK-100 root: `https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m`
    with paths like `P04/videos/P04_120.MP4`
  - EK-55 root: `https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d`
    with paths like `videos/train/P01/P01_09.MP4`
  `src/resolve_urls.py` maps every video_id to its exact URL using the official
  `md5.csv` from the download-scripts repo (path column + 55/100 tag).

## How the clips were obtained (bandwidth reality check)

- Full source videos are impractical here: P04_120 is 2,528,346,274 bytes for a
  835 s video (~24 Mbps GoPro originals), and measured throughput from
  data.bris.ac.uk on this VM is ~0.19 MB/s single-stream (~0.6 MB/s with 4
  parallel streams). One full video would take hours.
- Instead, each clip window is cut directly from the official file with ffmpeg
  over HTTPS (`-ss <clip_start> -i <official URL> -t <window>`), which uses HTTP
  range requests and transfers only the needed byte ranges plus the MP4 index.
  A 14 s clip took ~346 s wall-clock and produced a 9.2 MB file. Extraction runs
  4-way parallel (`src/extract_clips.py`) and is resumable per clip.
- Total local data stays far under the ~3 GB cap: measured after the build,
  `clips/` = 191 MB (25 clips) and `data/` = 89 MB (repos + annotations).

## Clip selection (reproducible: `src/select_segments.py`)

From EPIC_100_train + EPIC_100_validation (76,887 annotated segments):
- kept segments with duration 2.5–10 s, a real object noun (not hand/fingers),
  and a narration of ≤ 4 words (single clear action);
- clip window = action plus a per-clip pre-context of 3–8 s (deterministic hash
  of the narration id, so the action does NOT start at a constant offset —
  that would leak the answer to the grounding task) and 4 s after;
- windows within a video never overlap; round-robin across videos for kitchen
  diversity; at most 3 uses of any one verb.
- Result: **25 clips, 25 source videos, 17 distinct verbs** (`data/manifest.csv`).
  (26 were selected; EK26/P35_108 was dropped after its extraction stalled for
  12+ minutes on a 2.28 s seek — the file's index fetch over this slow link did
  not complete, and 25 clips is inside the 20–30 target. Every kept clip was
  re-verified with ffprobe against its manifest window after download.)

Each manifest row carries: clip id, source video, participant, action label
(narration), verb, noun, all annotated nouns, action start/end in the source
video, clip window in the source, **action start/end within the clip** (the
temporal-grounding ground truth), a factual question
("What object does the person interact with during the action '<narration>'?")
and its answer (the annotated noun; any of all_nouns accepted by the scorer).

## Known caveats

- Ground truth inherits EK-100 annotation properties: boundaries are the
  annotated narration segment boundaries, which are human-judged, not
  frame-exact. The analysis tolerance (±2 s) is set with this in mind.
- Base clips are standardized to 720p/30fps (sources are 1080p at 50/59.94 fps).
  Uniform across ALL conditions, so comparisons are unaffected; noted here
  because absolute grounding numbers might differ slightly at native fidelity.
