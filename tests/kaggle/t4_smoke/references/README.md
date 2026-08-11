# Committed metric references

Each file here is a per-step `loss` / `grad_norm` trace captured from one
real run, on one named GPU, with one named library set. `run_t4_smoke.py
--reference <file>` compares a fresh run against it.

## What this comparison is, and is not

It is a **band**, not an equality. Bitwise agreement across environments is
not achievable and is not claimed:

* fp16 reduction order changes with kernel selection, which changes with the
  driver, the CUDA runtime and the library versions.
* The T4 has no bf16, so the whole run is fp16 with a gradient scaler. Which
  steps the scaler skips is deterministic within an environment but can move
  between them, and a skipped step shifts every later value.
* Unsloth picks its attention backend by availability -- flash-attention,
  then xformers, then SDPA. A T4 resolves to xformers. A different backend
  is a different numeric path.

The exact assertion in this test is **run-to-run bitwise equality between
two fresh processes in the same session**. That one is exact, and it is what
catches nondeterminism. The reference band is the weaker, cross-environment
companion: it catches a change large enough to be a change in the
optimisation rather than in the low bits.

## Tolerance

Default `--rel-tol 0.10`, with `--abs-floor 0.05` on the denominator.

10% is chosen to sit above environment drift and below anything meaningful.
For scale, on the committed configuration the loss falls from about 10.3 to
about 0.14 across ten steps, so a genuine regression in the optimisation
moves a step by whole multiples, not by a few percent.

**The absolute floor is currently inert, and that is a measurement.** The
floor only engages where `|reference value| < 0.05`, and the smallest value
this ten-step trajectory produces is a loss of 0.1428, with the smallest
non-NaN grad_norm at 13.1. Nothing on the curve gets near 0.05, so
`max(|reference|, 0.05)` is just `|reference|` at every step, and removing
the floor entirely would not change a single verdict. It is retained as a
guard for a configuration that does drive the loss below 0.05 -- more steps,
a higher learning rate, an easier target -- not because this one does.
`test_whether_the_absolute_floor_is_reached_at_all` re-derives which of
those two worlds the committed reference lives in and asserts the floor
behaves accordingly, so the claim cannot quietly go stale.

Those numbers come from a B200 with `--force-sdpa`, which is evidence about
the shape of the trajectory and not about T4 numerics; the step at which a
T4 crosses 0.05, if it ever does, is unmeasured.

`NaN` grad_norm entries are expected, not corrupt: under fp16 the gradient
scaler reports NaN on any step whose gradients overflowed and then skips
that step. They are compared as NaN-equals-NaN.

## Current state: there is no committed reference

This directory holds no `.json` yet, so `check_reference` returns `absent`
and the band check is inert. That is deliberate rather than an oversight: a
valid reference has to come off a Kaggle T4, and no T4 run has yet got a
payload as far as producing metrics. The two tests that perturb the
committed reference skip until one exists, and they say so when they skip.

Do not fill this gap with numbers from other hardware. A trace captured on
any other card, or with `--force-sdpa`, is evidence about the harness and
not about T4 numerics, and committing it here would produce a check that
fails for a reason that is not a regression.

## Capturing one

The reference is a whole-file copy of the per-step trace a green run already
reports, so it does not need its own Kaggle session. From the
`kaggle-t4-evidence-<run id>` artifact of a run whose verdict was `pass`:

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

Regenerate only when a metric change has been understood and accepted, never
to silence a red run. The captured `environment` block travels with the file
and is echoed in the job summary, so a reader can always see which machine
the numbers came from.
