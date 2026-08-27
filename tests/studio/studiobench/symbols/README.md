# studiobench symbol bridges

Persisted `react-dom@<version>-<bundle-sha>.json` artefacts produced by
`analysis/bridge_build.py` and consumed by `analysis/symbols.py`.

## Why these files exist

React ships `cjs/react-dom-client.production.js` and
`cjs/react-dom-profiling.profiling.js` **already minified**, before Vite sees
them. Source maps and `keepNames` therefore recover our own component names and
recover nothing at all inside react-dom. The original identifier really is `Zk`.
No build flag changes that; it is not a misconfiguration.

A bridge is built by running the identical fixture at two or three small rungs
under `Profiler.startPreciseCoverage` against a **development** build and
against the shipping-shaped **profiling** build, then matching functions by
exact call-count vector. Invocation counts are a semantic invariant across build
modes, so a function with vector `(157, 1277, 879)` in dev is the same function
as the one with that vector in prod.

**The development build is used strictly as a dictionary and never as a
measurement.** `assert_no_measurements` refuses any float anywhere in these
files, and the schema has no duration fields at all, so there is nowhere a dev
millisecond could sit even by accident.

## What is in a file

| field | meaning |
|---|---|
| `status` | `ok`, or `failed` with `failure_reason` |
| `react_version`, `bundle_sha` | a bridge is valid only for the exact bytes it was built against; a rebuild renames everything |
| `rungs` | the ladder both builds ran, in order |
| `mapping` | `<bundle basename>:<startOffset>:<endOffset>` to a real function name |
| `evidence` | the count vector that produced each match, for auditing |
| `ambiguous_prod` / `ambiguous_dev` | vectors shared by more than one function in their own build. **Never guessed at**, only reported |
| `anchors_checked`, `anchor_failures` | our own components, independently named on both sides, which must map to themselves |

## The failure mode this is designed around

If any anchor fails, the **whole bridge is discarded**, not just the bad anchor,
and the run degrades to unnamed frames with `symbol_bridge: failed`. A bridge
that mislabels a function it can check will mislabel functions it cannot, and an
unnamed frame is a much smaller problem than a confidently wrong name.

## Observed coverage

Against a real React 19.2.4 app on a real Vite 8 profiling build, one bridge
resolved 19 functions from 26 unique-vector candidates, with 25 prod and 29 dev
vectors declared ambiguous and left alone. Among the resolved names were
`createWorkInProgress`, `bailoutOnAlreadyFinishedWork`, `completeUnitOfWork`,
`updateSlot` and `useFiber`, which is to say the exact fiber-bookkeeping
functions the M1 hypothesis is about.

Expect roughly this: a minority of react-dom resolves, heavily biased towards
the functions that run a lot, which are also the ones worth naming. Functions
that run once or twice share vectors with everything and are correctly refused.
