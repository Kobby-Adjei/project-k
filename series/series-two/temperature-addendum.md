# Series Two addendum: the temperature check (Nov 5, 2026)

The 87-ask battery is locked and untouched. This addendum
answers one later question about it: every spoken answer in
the project was taken at temperature 0 (Pegasus) or greedy
(open models); would a warm draw change the answers?

- The Pegasus arm ran Nov 5, 2026: temperature 0.7, 3 draws
  per cell, 30 calls, the Pilot Two denial cells (swap files
  under P3, three real clips under P1), prompts byte-identical.
  Result: every registered prediction held. Non-speech sounds
  reported 0 of 3 in every cell; real clips kept heard "none"
  3 of 3; controls (silence, speech) held. The spoken-answer
  findings are not a temperature-0 artifact at 0.7.
- The Qwen 3B part (greedy anchor plus 0.7 and 1.0 draws on
  the speech and blackout cells) was built and parked by
  Kobby's ruling: the temperature doubt bites on Pegasus,
  whose claims rest on spoken answers. Notebook on Drive,
  unrun to completion.

Full plan, results and read:
~/workspace/project-k-pilot/temperature-check/ (plan.md,
tempcheck_pegasus_results.csv, pegasus-arm-read.md).
