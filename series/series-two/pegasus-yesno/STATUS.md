# Pegasus yes/no: status

The text version of the Series Two yes or no battery for
Pegasus 1.6: 30 cells, 2 repeats, 60 calls through the
Twelve Labs Analyze route, temperature 0. The runner is
run_pegasus_yesno.py in this folder; it is resume-safe and
records failures instead of stopping.

State as of Nov 2, 2026: 14 of 60 calls answered. The run
then hit the account's workload allowance. The provider's
error, verbatim: code "workload_allowance_exhausted",
message "The Workload allowance is exhausted and overage
is disabled." 22 recorded cells carry that failure and 24
were never attempted.

The run resumes when the allowance resets. No second
account will be created to get around the limit; the run
finishes on the account that started it, or it waits.

Completion (Nov 5, 2026): that last rule was superseded by
the account owner's own decision. The remaining 46 calls
ran on his other, pre-existing Twelve Labs account (held
under his personal Gmail; it was not created for this) and
finished the battery: 60 of 60 answers in, all HTTP 200.
Results: pegasus_yesno_results.csv in this folder; the
read: pegasus-yesno-diagnostics-summary.md. In short: 40
of 60 correct, and all 20 wrong answers are one kind — a
non-speech sound present in the clip, answered No. Series
Two is complete for all five models.
