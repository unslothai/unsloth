# Committed metric references

Each file here is a per-step `loss` / `grad_norm` trace captured from one
real run, on one named GPU, with one named library set. `run_t4_smoke.py
--reference <file>` compares a fresh run against it.

## Which leg this applies to, and which it must not

Only the **control** leg. It installs the pinned library set in
`../pins/control.txt`, which is the set the committed trace was captured
with, so a band comparison against it is a comparison of one environment
with itself.

The **canary** leg runs the identical payload on the newest library set
Unsloth's constraints allow, and it is deliberately given **no reference at
all** (`legs.py`, `LEGS["canary"].reference == ""`). Two library versions do
not produce one fp16 trajectory: kernel selection moves with every one of
them, and a single moved gradient-scaler skip shifts every later step. A
band check there would go red on ordinary cross-version drift, which is
exactly the kind of red that gets a check switched off before the day it is
right.

So the canary asserts what does not depend on versions, and all of it is
already in the payload: run-to-run bitwise equality between its own two
fresh processes, the exact canary string, that the optimizer applied
updates at all, and that nothing raised. The one thing it adds over the
control is the resolved version of every watched package, recorded in its
report so that "canary red, control green" is a bisect and not an
investigation.

## What this comparison is, and is not

It is a **band**, not an equality. Bitwise agreement across environments is
not achievable and is not claimed:

* fp16 reduction order changes with kernel selection, which changes with the
  driver, the CUDA runtime and the library versions.
* The T4 has no bf16, so the whole run is fp16 with a gradient scaler. Which
  steps the scaler skips is deterministic within an environment but can move
  between them, and a skipped step shifts every later value.
* Unsloth picks its attention backend by availability -- flash-attention,
  then xformers, then SDPA. A different backend is a different numeric path.
  **On the Kaggle session this reference came from, xformers was not
  installed at all**: the banner read `Bfloat16 = FALSE. FA [Xformers =
  None. FA2 = False]`, because `unsloth` goes in with `--no-deps`,
  `unsloth_zoo`'s dependency set does not carry xformers and the Kaggle
  image does not either. So this trace is the fallback path, not the
  xformers path. If xformers is ever added to the install, these numbers
  move and the reference must be recaptured.

The exact assertion in this test is **run-to-run bitwise equality between
two fresh processes in the same session**. That one is exact, and it is what
catches nondeterminism. The reference band is the weaker, cross-environment
companion: it catches a change large enough to be a change in the
optimisation rather than in the low bits.

## The step count is part of the reference

A reference is a trace of one run, and a run of a different length is a
different run. Step 4 of a 10-step run and step 4 of a 3-step run are the
same iterate only by coincidence: the fp16 scaler spends the front of a run
discovering its loss scale, and a short run is all front.

So every reference records the `max_steps` it was captured at, in its
`config` block, and `check_reference` compares that against the step count
of the run in hand **before** it looks at a single number. A mismatch is
status `step_count_mismatch` and a hard failure. A reference with no
recorded step count is status `reference_step_count_unknown` and also a hard
failure -- "it does not say" is not "it matches".

That refusal is deliberately louder than a skip. A band check quietly
comparing a 3-step run against a 10-step trace would either fail for the
wrong reason or, worse, pass on the first three steps of a curve it has
nothing to do with, and report green while checking nothing.

Changing `--max-steps`, `--init-loss-scale`, the learning rate, the model or
the optimizer therefore invalidates the committed file. There is no
tolerance that covers it and widening the band is never the answer.

## Tolerance

Default `--rel-tol 0.10`, with `--abs-floor 0.05` on the denominator.

10% is chosen to sit above environment drift and below anything meaningful.
For scale, on the configuration the committed file was captured with the
loss falls from about 10.3 to about 0.09 across ten steps, so a genuine
regression in the optimisation moves a step by whole multiples, not by a few
percent. Every measurement in the two sections below is likewise about that
ten-step capture, which is the only trace this directory has so far.

**The absolute floor is currently inert, and that is a measurement.** The
floor only engages where `|reference value| < 0.05`, and the smallest value
this ten-step trajectory produces is a loss of 0.0871, with the smallest
non-NaN grad_norm at 11.2. Nothing on the curve gets near 0.05, so
`max(|reference|, 0.05)` is just `|reference|` at every step, and removing
the floor entirely would not change a single verdict. It is retained as a
guard for a configuration that does drive the loss below 0.05 -- more steps,
a higher learning rate, an easier target -- not because this one does.
`test_whether_the_absolute_floor_is_reached_at_all` re-derives which of
those two worlds the committed reference lives in and asserts the floor
behaves accordingly, so the claim cannot quietly go stale.

Those numbers are the committed T4 trace itself. The final loss, 0.0871, is
under a factor of two above the floor, so a configuration change that pushes
the trajectory a little further would engage it;
`test_whether_the_absolute_floor_is_reached_at_all` re-derives which world
the committed file is in on every run rather than trusting this paragraph.

`NaN` grad_norm entries are expected, not corrupt: under fp16 the gradient
scaler reports NaN on any step whose gradients overflowed and then skips
that step. They are compared as NaN-equals-NaN.

## Current state: CURRENT

`t4_qwen2.5-0.5b.json` was captured at **`max_steps=10`**, which is what the
workflow runs, so it applies as committed and no recapture is pending.

Shortening the run was considered and rejected on this file's own evidence.
The scaler skips steps 1, 2 and 3, so a 3-step run applies zero optimizer
updates: the loss stays around 10, the canary never forms, and the band
would compare three points of a curve that never moved. Nor would it save
meaningful quota, since a launch costs about 0.08h and that is dominated by
pip install rather than by training. `--init-loss-scale` exists for anyone
who does want a shorter run; it changes the numeric path, so it requires a
reference recaptured with it, which `check_reference` enforces rather than
trusts. See "Recapturing after a configuration change".

What this file documents, from kernel
`danielhanchen/unsloth-t4-ci-e3c6661f`, the first green run this workflow
has had. What that run measured, on real `Tesla T4` / `sm_75` / 14.6 GB
hardware:

* Both payloads, one per T4 of the session, passed every assertion.
* Two fresh processes agreed **bitwise** on all ten steps, on both cards:
  `max_abs_diff` was exactly `0.0` for loss and for grad_norm.
* The two cards, independently, produced the **same ten values to the last
  bit** as each other. That is four processes agreeing, not two.
* All four cycles emitted the canary `__UNSLOTH__!!!` exactly.
* The fp16 scaler skipped steps 1, 2 and 3 on every cycle.

The committed file is `reports[0]` of that run, per the recipe below. Peak
reserved memory was 0.7 GB per payload and each cycle trained in 15-26 s.

Do not fill this file with numbers from other hardware. A trace captured on
any other card, or with `--force-sdpa`, is evidence about the harness and
not about T4 numerics, and committing it here would produce a check that
fails for a reason that is not a regression.

## Capturing one

The reference is a whole-file copy of the per-step trace a green run already
reports, so it does not need its own Kaggle session. From the
`kaggle-t4-evidence` artifact of a run whose verdict was `pass`:

```
python - <<'PY'
import json, pathlib
result = json.loads(pathlib.Path("kaggle_evidence/launch_result.json").read_text())
report = result["reports"][0]
pathlib.Path("tests/kaggle/t4_smoke/references/t4_qwen2.5-0.5b.json").write_text(
    json.dumps({"metrics": report["metrics"],
                "environment": report["environment"],
                "config": report["config"],
                "source_kernel": result["slug"]}, indent=2))
PY
```

Take it from `reports[0]`, whose metrics are cycle 0 of the payload that
passed. Both payloads in a session run the same configuration on the two
T4s of that session; if their traces disagree, that disagreement is itself
the finding and nothing should be committed until it is understood.

`json.dumps` writes `NaN` for the scaler-skipped steps. That is not valid
strict JSON but Python's `json.loads` reads it back, which is what both the
payload and the launcher use, and the alternative -- dropping those entries
-- would silently exempt exactly the steps whose behaviour matters most.

`config` is not optional. It carries the `max_steps` the trace was captured
at, and without it every future comparison refuses with
`reference_step_count_unknown`. Copy the block through verbatim; do not
hand-edit `max_steps` to match a run it did not come from.

Regenerate only when a metric change has been understood and accepted, never
to silence a red run. The captured `environment` block travels with the file
and is echoed in the job summary, so a reader can always see which machine
the numbers came from.

## Recapturing after a configuration change

A change to `--max-steps`, `--init-loss-scale`, the learning rate, the model
or the optimizer means the committed reference no longer describes the run.
The band check then refuses, loudly, on every run. Clearing that takes
exactly one Kaggle session:

1. Dispatch **Kaggle T4 Notebook CI** with `skip_reference_band: true`. That
   is the only supported way to run without a band check, it prints a
   `::warning` saying so, and everything else still applies: the bitwise
   run-to-run check, the canary, and the "did the optimizer actually update
   anything" check all still gate the run. If the new configuration cannot
   pass those, it has no business becoming a reference.
2. Leave `max_steps` (and any other input) at the value the workflow will
   use from then on. A reference captured at a step count nobody runs is
   worth nothing.
3. Download the `kaggle-t4-evidence` artifact from that run and apply the
   recipe above.
4. Commit the new file with the kernel slug in the message, and check the
   diff: `config.max_steps` must be the new count, and `environment` must
   still say `Tesla T4` / `sm_75`.

The run in step 1 is also the first hardware evidence for the new
configuration. If the canary goes missing at a shorter step count, or the
report says the loss-scale pin did not apply, that is the answer to "is this
configuration viable" and the step count goes back up rather than the
assertions coming down.
