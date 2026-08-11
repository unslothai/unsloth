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
moves a step by whole multiples, not by a few percent. The absolute floor
exists because the late steps approach zero, where an utterly harmless
absolute drift of 0.01 would otherwise register as a several-hundred-percent
relative deviation.

`NaN` grad_norm entries are expected, not corrupt: under fp16 the gradient
scaler reports NaN on any step whose gradients overflowed and then skips
that step. They are compared as NaN-equals-NaN.

## Refreshing a reference

Regenerate only when a metric change has been understood and accepted, never
to silence a red run:

```
python tests/kaggle/t4_smoke/run_t4_smoke.py --outdir /tmp/ref --repeat 1 --cycle 0
cp /tmp/ref/cycle_report.json tests/kaggle/t4_smoke/references/<name>.json
```

The captured `environment` block travels with the file and is echoed in the
job summary, so a reader can always see which machine the numbers came from.
A reference captured anywhere other than a Kaggle T4 is not a valid
reference for this test.
