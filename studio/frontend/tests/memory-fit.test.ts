// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Load-Model memory row's verdicts and the single note it prints.
//
// This file exists because the chain used to live inside model-config-page.tsx, where
// the node runner cannot reach it: a 3,400-line .tsx that imports React, the router
// and thirty component barrels. An arm that could never be taken shipped in it, and
// nothing could have caught that except reading the ternary carefully enough.

import assert from "node:assert/strict";
import test from "node:test";
import type { SystemGpuDevice } from "../src/hooks/gpu-selection.ts";
import { aggregateVramReserveDeficitGb } from "../src/hooks/gpu-vram.ts";

import {
  type MemoryFitCapacity,
  type MemoryFitEstimate,
  type MemoryFitVerdict,
  classifyMemoryFit,
  formatMemoryGb,
  memoryFigureCandidates,
  resolveDraftCacheNote,
  resolveKvNote,
  resolveMemoryFit,
  resolveReclaimableMemoryCredit,
  worseMemoryFit,
} from "../src/features/model-picker/model-config/memory-fit.ts";

const GB = 1024 ** 3;

const SIZED: MemoryFitEstimate = {
  gpuBytes: 0,
  totalBytes: 0,
  kvEstimable: true,
  drafterKvUnsized: false,
  adaptersUnsized: false,
  moeOffloadUnmodelled: false,
};

/** A roomy discrete host: 24 GB card, 64 GB of RAM, nothing else running. */
const IDLE_DISCRETE: MemoryFitCapacity = {
  gpuCapacityGb: 24,
  totalCapacityGb: 88,
  systemRamCapacityGb: 64,
  freeGpuCapacityGb: 23,
  usableSystemRamGb: 60,
  singleMemoryPool: false,
};

/** Apple, 64 GB unified. One pool for everything. */
const APPLE: MemoryFitCapacity = {
  gpuCapacityGb: 64,
  totalCapacityGb: 64,
  systemRamCapacityGb: 64,
  freeGpuCapacityGb: 60,
  usableSystemRamGb: 60,
  singleMemoryPool: true,
};

const fit = (
  estimate: Partial<MemoryFitEstimate>,
  capacity: Partial<MemoryFitCapacity>,
  base: MemoryFitCapacity = IDLE_DISCRETE,
) => resolveMemoryFit({ ...SIZED, ...estimate }, { ...base, ...capacity });

// ---------------------------------------------------------------------------
// classifyMemoryFit

test("a footprint well inside the capacity fits", () => {
  assert.equal(classifyMemoryFit(8 * GB, 24), "fits");
});

test("above 85% of the capacity is tight, above 100% exceeds", () => {
  assert.equal(classifyMemoryFit(20.5 * GB, 24), "tight");
  assert.equal(classifyMemoryFit(25 * GB, 24), "exceeds");
  // The boundary itself is not tight: 85% exactly is still a fit.
  assert.equal(classifyMemoryFit(0.85 * 24 * GB, 24), "fits");
  // And a footprint exactly filling the capacity has not exceeded it.
  assert.equal(classifyMemoryFit(24 * GB, 24), "tight");
});

test("nothing probed and nothing to weigh are both no verdict", () => {
  assert.equal(classifyMemoryFit(8 * GB, 0), "unknown");
  assert.equal(classifyMemoryFit(0, 24), "unknown");
  assert.equal(classifyMemoryFit(8 * GB, -4), "unknown");
  assert.equal(classifyMemoryFit(-8 * GB, 24), "unknown");
});

// The regression this guard exists for. Every one of these came back "fits" before:
// NaN and Infinity fail `<= 0` and every ratio comparison, so control fell all the way
// through to the confident green answer at the bottom.
test("a non-finite reading is never a fit", () => {
  for (const bytes of [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]) {
    assert.equal(classifyMemoryFit(bytes, 24), "unknown", `bytes=${bytes}`);
  }
  for (const capacity of [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]) {
    assert.equal(classifyMemoryFit(8 * GB, capacity), "unknown", `capacity=${capacity}`);
  }
});

test("a value that is not a number at all is not a fit either", () => {
  const bad = [null, undefined, "24", "", {}, []] as unknown as number[];
  for (const value of bad) {
    assert.equal(classifyMemoryFit(value, 24), "unknown", `bytes=${String(value)}`);
    assert.equal(classifyMemoryFit(8 * GB, value), "unknown", `capacity=${String(value)}`);
  }
});

// ---------------------------------------------------------------------------
// worseMemoryFit

test("the worse of two verdicts wins, and unknown loses to any real one", () => {
  assert.equal(worseMemoryFit("fits", "exceeds"), "exceeds");
  assert.equal(worseMemoryFit("exceeds", "fits"), "exceeds");
  assert.equal(worseMemoryFit("tight", "fits"), "tight");
  assert.equal(worseMemoryFit("unknown", "fits"), "fits");
  assert.equal(worseMemoryFit("fits", "unknown"), "fits");
  assert.equal(worseMemoryFit("unknown", "unknown"), "unknown");
});

test("worseMemoryFit is symmetric for every pair", () => {
  const all: MemoryFitVerdict[] = ["unknown", "fits", "tight", "exceeds"];
  for (const a of all) {
    for (const b of all) {
      assert.equal(worseMemoryFit(a, b), worseMemoryFit(b, a), `${a} vs ${b}`);
    }
  }
});

// ---------------------------------------------------------------------------
// D1: the advisory chain
//
// The defect: the pool ternary gated the WHOLE tail. Inside its
// `singleMemoryPool === false` arm, the host-pressure branch chose its wording by
// re-testing `singleMemoryPool` -- which is false there by construction -- so the
// single-pool string could not be selected by any input. The visible half of the same
// bug is that a single-pool host had exactly one reachable note.

const ADVISORY_TEXTS = {
  singlePoolExceeds:
    "Exceeds shared memory. Try a shorter context or smaller model; CPU offloading adds no memory.",
  singlePoolPressure:
    "Fits this machine, but little memory is free right now. Free memory or try Auto context.",
  hostShareExceeds:
    "CPU placement exceeds system RAM. Try fewer CPU layers or a smaller model.",
  totalExceeds:
    "Exceeds combined GPU and system memory. Try a shorter context or smaller model.",
  gpuExceeds:
    "Exceeds GPU memory. Try Auto context or fewer GPU layers; loading may still fail.",
  hostPressure:
    "Fits system RAM, but little is free right now. Free memory or use fewer CPU layers.",
  gpuPressure:
    "Fits this GPU, but little VRAM is free right now. Free memory or try Auto context.",
};

test("D1: a single-pool host under memory pressure now says so", () => {
  // Apple, 64 GB unified, a 30 GB load, 6 GB actually free. The machine holds it;
  // what is free does not. Before the fix this produced no note whatsoever, because
  // the only branch a single pool could reach was "exceeds".
  const result = fit(
    { gpuBytes: 30 * GB, totalBytes: 30 * GB },
    { freeGpuCapacityGb: 6, usableSystemRamGb: 6 },
    APPLE,
  );
  assert.equal(result.totalFit, "fits");
  assert.ok(result.gpuPressured || result.hostPressured);
  assert.deepEqual(result.advisory, {
    tone: "muted",
    text: ADVISORY_TEXTS.singlePoolPressure,
  });
});

test("D1: the single-pool pressure text is reachable from EITHER free reading", () => {
  // Free VRAM sees it, host RAM does not.
  const gpuSide = fit(
    { gpuBytes: 30 * GB, totalBytes: 30 * GB },
    { freeGpuCapacityGb: 6, usableSystemRamGb: 60 },
    APPLE,
  );
  assert.equal(gpuSide.advisory?.text, ADVISORY_TEXTS.singlePoolPressure);
  // Host RAM sees it, free VRAM does not.
  const hostSide = fit(
    { gpuBytes: 30 * GB, totalBytes: 30 * GB },
    { freeGpuCapacityGb: 60, usableSystemRamGb: 6 },
    APPLE,
  );
  assert.equal(hostSide.advisory?.text, ADVISORY_TEXTS.singlePoolPressure);
});

test("a tight reading warns without claiming the load exceeds available memory", () => {
  const result = fit(
    { gpuBytes: 26 * GB, totalBytes: 26 * GB },
    { freeGpuCapacityGb: 30, usableSystemRamGb: 30 },
    APPLE,
  );
  assert.equal(result.freeGpuFit, "tight");
  assert.equal(result.usableHostFit, "tight");
  assert.match(result.advisory?.text ?? "", /little memory is free right now/);
  assert.doesNotMatch(result.advisory?.text ?? "", /not what is free|will be|refused/);
});

test("D1: no discrete-host string can be chosen on a single-pool host", () => {
  // The other half of the dead branch: the two discrete pressure strings talk about
  // "the card" and "the part of this load that runs from system RAM", neither of
  // which means anything where there is one pool.
  const sweep = new Set<string>();
  for (const gpuBytes of [0, 4 * GB, 30 * GB, 90 * GB]) {
    for (const totalBytes of [0, 4 * GB, 30 * GB, 90 * GB]) {
      for (const freeGpuCapacityGb of [0, 2, 6, 60]) {
        for (const usableSystemRamGb of [0, 2, 6, 60]) {
          const note = fit(
            { gpuBytes, totalBytes },
            { freeGpuCapacityGb, usableSystemRamGb },
            APPLE,
          ).advisory?.text;
          if (note) sweep.add(note);
        }
      }
    }
  }
  assert.deepEqual(
    [...sweep].sort(),
    [ADVISORY_TEXTS.singlePoolExceeds, ADVISORY_TEXTS.singlePoolPressure].sort(),
  );
});

test("D1: the discrete host keeps its own wording, and never the single-pool one", () => {
  const sweep = new Set<string>();
  for (const gpuBytes of [0, 4 * GB, 20 * GB, 30 * GB]) {
    for (const totalBytes of [0, 4 * GB, 30 * GB, 200 * GB]) {
      for (const freeGpuCapacityGb of [0, 2, 23]) {
        for (const usableSystemRamGb of [0, 2, 60]) {
          const note = fit(
            { gpuBytes, totalBytes },
            { freeGpuCapacityGb, usableSystemRamGb },
          ).advisory?.text;
          if (note) sweep.add(note);
        }
      }
    }
  }
  assert.equal(sweep.has(ADVISORY_TEXTS.singlePoolPressure), false);
  assert.equal(sweep.has(ADVISORY_TEXTS.singlePoolExceeds), false);
});

test("the floor notes outrank every verdict, in their own order", () => {
  // An unsizable cache says the figures are incomplete, which beats any reading of
  // them. The header case outranks the drafter case, and both outrank the MoE note.
  const both = fit(
    { kvEstimable: false, drafterKvUnsized: true, moeOffloadUnmodelled: true, totalBytes: 900 * GB },
    {},
  );
  assert.equal(both.advisory?.tone, "warn");
  assert.match(both.advisory?.text ?? "", /attention dimensions/);
  const drafter = fit(
    { drafterKvUnsized: true, moeOffloadUnmodelled: true, totalBytes: 900 * GB },
    {},
  );
  assert.match(drafter.advisory?.text ?? "", /remote draft model or vision component/);
  const moe = fit({ moeOffloadUnmodelled: true, gpuBytes: 8 * GB, totalBytes: 900 * GB }, {});
  assert.equal(moe.advisory?.tone, "muted");
  assert.match(moe.advisory?.text ?? "", /Expert layers/);
});

test("the floor marker follows either unsizable case", () => {
  assert.equal(fit({ kvEstimable: false }, {}).prefix, "≥ ");
  assert.equal(fit({ drafterKvUnsized: true }, {}).prefix, "≥ ");
  assert.equal(fit({}, {}).prefix, "");
  assert.equal(fit({ kvEstimable: false }, {}).bounded, true);
  assert.equal(fit({}, {}).bounded, false);
});

test("the aggregate verdict is asked before the GPU one", () => {
  // A 200 GB load on a 24 GB card and 64 GB of RAM. Reading gpuFit alone offered
  // spilling to system RAM as the remedy, which is advice to do something that cannot
  // work.
  const result = fit({ gpuBytes: 20 * GB, totalBytes: 200 * GB }, {});
  assert.equal(result.advisory?.text, ADVISORY_TEXTS.totalExceeds);
});

test("both pools overflowing never recommends moving more layers to the GPU", () => {
  for (const gpuGb of [24, 30]) {
    const result = fit({ gpuBytes: gpuGb * GB, totalBytes: (gpuGb + 70) * GB }, {});
    assert.equal(result.advisory?.text, ADVISORY_TEXTS.totalExceeds);
    assert.doesNotMatch(result.advisory?.text ?? "", /fewer CPU layers/);
  }
});

test("host overflow with spare GPU capacity keeps placement advice", () => {
  const result = fit({ gpuBytes: 8 * GB, totalBytes: 78 * GB }, {});
  assert.equal(result.advisory?.text, ADVISORY_TEXTS.hostShareExceeds);
});

test("a load beyond GPU and RAM combined, with a host share that fits RAM", () => {
  const result = fit(
    { gpuBytes: 80 * GB, totalBytes: 120 * GB },
    { gpuCapacityGb: 24, totalCapacityGb: 88, systemRamCapacityGb: 64 },
  );
  assert.equal(result.advisory?.text, ADVISORY_TEXTS.totalExceeds);
});

test("a load that only overflows the card gets conditional offload advice", () => {
  const result = fit({ gpuBytes: 30 * GB, totalBytes: 30 * GB }, {});
  assert.equal(result.advisory?.text, ADVISORY_TEXTS.gpuExceeds);
});

test("a discrete host under host-RAM pressure keeps the system-RAM wording", () => {
  const result = fit(
    { gpuBytes: 10 * GB, totalBytes: 50 * GB },
    { usableSystemRamGb: 38 },
  );
  assert.equal(result.advisory?.text, ADVISORY_TEXTS.hostPressure);
});

test("a discrete host under VRAM pressure alone gets the card wording", () => {
  const result = fit(
    { gpuBytes: 20 * GB, totalBytes: 20 * GB },
    { freeGpuCapacityGb: 8 },
  );
  assert.equal(result.advisory?.text, ADVISORY_TEXTS.gpuPressure);
  // And the GPU figure is coloured amber rather than left green.
  assert.equal(result.rawGpuFit, "fits");
  assert.equal(result.gpuFit, "tight");
});

test("a comfortable load says nothing at all", () => {
  assert.equal(fit({ gpuBytes: 6 * GB, totalBytes: 6 * GB }, {}).advisory, null);
  assert.equal(
    fit({ gpuBytes: 6 * GB, totalBytes: 6 * GB }, {}, APPLE).advisory,
    null,
  );
});

// ---------------------------------------------------------------------------
// The pool-aware free reading

test("one pool weighs the WHOLE load against what is free, not the GPU share", () => {
  // A shared iGPU with half the load CPU-offloaded. Those bytes come out of the same
  // memory, so measuring only the GPU share against free memory called a load that
  // cannot fit comfortable.
  const pooled = fit(
    { gpuBytes: 6 * GB, totalBytes: 30 * GB },
    { freeGpuCapacityGb: 10, usableSystemRamGb: 10 },
    { ...APPLE, singleMemoryPool: true },
  );
  assert.equal(pooled.freeGpuFit, "exceeds");
  assert.equal(pooled.gpuPressured, true);
  // The discrete host asks the same question of the GPU share alone, which is right
  // there: the host bytes are a different pool.
  const discrete = fit(
    { gpuBytes: 6 * GB, totalBytes: 30 * GB },
    { freeGpuCapacityGb: 10 },
  );
  assert.equal(discrete.freeGpuFit, "fits");
});

test("the host share is the bytes outside the GPU, floored at zero", () => {
  assert.equal(fit({ gpuBytes: 10 * GB, totalBytes: 30 * GB }, {}).hostShareBytes, 20 * GB);
  // gpu_bytes above total_bytes is nonsense off the wire, not a negative footprint.
  assert.equal(fit({ gpuBytes: 30 * GB, totalBytes: 10 * GB }, {}).hostShareBytes, 0);
});

test("one pool asks no separate host-share question", () => {
  // The combined figure already describes that case exactly, so a second verdict
  // drawn from the same bytes would only be able to disagree with it.
  assert.equal(fit({ gpuBytes: 6 * GB, totalBytes: 30 * GB }, {}, APPLE).hostShareFit, "unknown");
});

// ---------------------------------------------------------------------------
// Garbage in

test("a non-finite footprint produces no verdict and no advisory, and does not throw", () => {
  for (const bad of [Number.NaN, Number.POSITIVE_INFINITY]) {
    const result = fit({ gpuBytes: bad, totalBytes: bad }, {});
    assert.equal(result.gpuFit, "unknown");
    assert.equal(result.totalFit, "unknown");
    assert.equal(result.advisory, null);
    // A number the caller can print. Math.max(0, NaN) is NaN, not 0.
    assert.equal(result.hostShareBytes, 0);
  }
});

test("a non-finite capacity never yields a fit", () => {
  const result = fit(
    { gpuBytes: 8 * GB, totalBytes: 8 * GB },
    {
      gpuCapacityGb: Number.NaN,
      totalCapacityGb: Number.POSITIVE_INFINITY,
      systemRamCapacityGb: Number.NaN,
      freeGpuCapacityGb: Number.NaN,
      usableSystemRamGb: Number.POSITIVE_INFINITY,
    },
  );
  assert.equal(result.gpuFit, "unknown");
  assert.equal(result.totalFit, "unknown");
  assert.equal(result.gpuPressured, false);
  assert.equal(result.hostPressured, false);
});

test("no combination of garbage throws or produces a verdict outside the four", () => {
  const values = [
    0, -1, 1, Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY,
    null, undefined, "12", {},
  ] as unknown as number[];
  const allowed = new Set(["fits", "tight", "exceeds", "unknown"]);
  for (const a of values) {
    for (const b of values) {
      const result = fit(
        { gpuBytes: a, totalBytes: b },
        {
          gpuCapacityGb: a,
          totalCapacityGb: b,
          systemRamCapacityGb: a,
          freeGpuCapacityGb: b,
          usableSystemRamGb: a,
        },
      );
      for (const verdict of [
        result.gpuFit, result.rawGpuFit, result.totalFit,
        result.hostShareFit, result.freeGpuFit, result.usableHostFit,
      ]) {
        assert.ok(allowed.has(verdict), `${String(a)}/${String(b)} -> ${verdict}`);
      }
      // Never a confident green from a number that does not exist. `a` drives both
      // gpuBytes and gpuCapacityGb, so it is the one this verdict depends on; a bad
      // `b` only reaches the free reading, and losing that warning is not a false fit.
      if (!Number.isFinite(a)) {
        assert.notEqual(result.gpuFit, "fits", `gpuBytes=${String(a)}`);
      }
      if (!Number.isFinite(b)) {
        assert.notEqual(result.totalFit, "fits", `totalBytes=${String(b)}`);
      }
      // And every figure the row prints stays printable.
      assert.ok(
        Number.isFinite(result.hostShareBytes) && result.hostShareBytes >= 0,
        `hostShareBytes=${result.hostShareBytes}`,
      );
    }
  }
});

// ---------------------------------------------------------------------------
// formatMemoryGb

test("a figure is always finite and never negative", () => {
  // GiB, not GB. The divide was always by 1024**3, so every figure this printed
  // was a gibibyte value wearing a gigabyte label -- 7.4% high, on seven figures
  // of the Load Model panel. Same defect #9570 fixed elsewhere; the guard test
  // could not see this one because its regex only matches interpolations naming
  // a `*TotalGb`.
  assert.equal(formatMemoryGb(24 * GB), "24.00 GiB");
  assert.equal(formatMemoryGb(0), "0.00 GiB");
  assert.equal(formatMemoryGb(-5 * GB), "0.00 GiB");
  assert.equal(formatMemoryGb(Number.NaN), "0.00 GiB");
  assert.equal(formatMemoryGb(Number.POSITIVE_INFINITY), "0.00 GiB");
  assert.equal(formatMemoryGb(undefined as unknown as number), "0.00 GiB");
});

// ---------------------------------------------------------------------------
// The two captions

test("the KV caption names the dtype, what was priced, and where it lives", () => {
  assert.equal(
    resolveKvNote({ cacheTypeKv: "q8_0", nCtx: 32768, nParallel: 1, kvOnGpu: true }),
    "q8_0 · 32,768 tokens",
  );
  // No dtype reported falls back to f16, several slots are named, and a cache the
  // loader moved off the GPU says where it went.
  assert.equal(
    resolveKvNote({ cacheTypeKv: null, nCtx: 4096, nParallel: 4, kvOnGpu: false }),
    "f16 · 4,096 tokens · 4 slots · host RAM",
  );
  // A single slot is the unremarkable case and is not named.
  assert.equal(
    resolveKvNote({ cacheTypeKv: "f16", nCtx: 4096, nParallel: 1, kvOnGpu: false }),
    "f16 · 4,096 tokens · host RAM",
  );
});

test("the KV caption survives a field that is not a number", () => {
  // `.toLocaleString()` on a null throws, and one bad field must not take the panel
  // down with it.
  assert.doesNotThrow(() =>
    resolveKvNote({
      cacheTypeKv: null,
      nCtx: null as unknown as number,
      nParallel: Number.NaN,
      kvOnGpu: true,
    }),
  );
  assert.equal(
    resolveKvNote({
      cacheTypeKv: null,
      nCtx: Number.NaN,
      nParallel: 1,
      kvOnGpu: true,
    }),
    "f16 · 0 tokens",
  );
});

test("the draft cache note reads its OWN placement, not the target cache's", () => {
  // --spec-draft-ngl 0 moves the drafter while --no-kv-offload moves the target, so a
  // boolean read off the target was wrong in both directions.
  assert.equal(resolveDraftCacheNote(0, 4 * GB), "host RAM");
  // Under MTP the term is split across both placements: a third case.
  assert.equal(resolveDraftCacheNote(1 * GB, 4 * GB), "1.00 GiB on GPU");
  // Entirely on the GPU is the unremarkable case and gets no caption.
  assert.equal(resolveDraftCacheNote(4 * GB, 4 * GB), undefined);
  assert.equal(resolveDraftCacheNote(Number.NaN, 4 * GB), "host RAM");
});

test("an unsizable pass-through adapter marks the total a floor", () => {
  // llama.cpp loads every --lora / --control-vector into resident tensors on top of
  // the base model. When one is named but cannot be stat'd its bytes are missing, so
  // the figure is a lower bound and has to say so, exactly as an unsized drafter does.
  const bounded = resolveMemoryFit(
    { ...SIZED, adaptersUnsized: true, totalBytes: 8 * GB, gpuBytes: 8 * GB },
    IDLE_DISCRETE,
  );
  assert.equal(bounded.bounded, true);
  assert.equal(bounded.prefix, "≥ ");
  assert.equal(bounded.advisory?.tone, "warn");
  assert.match(bounded.advisory?.text ?? "", /adapter or control vector/);
  const sized = resolveMemoryFit(
    { ...SIZED, totalBytes: 8 * GB, gpuBytes: 8 * GB },
    IDLE_DISCRETE,
  );
  assert.equal(sized.bounded, false);
});

test("an unsized adapter warning takes precedence over a placement verdict", () => {
  const result = fit(
    { adaptersUnsized: true, moeOffloadUnmodelled: true, totalBytes: 200 * GB },
    {},
  );
  assert.match(result.advisory?.text ?? "", /adapter or control vector/);
  assert.equal(result.advisory?.tone, "warn");
});

test("CPU-only estimates use RAM guidance without suggesting GPU placement", () => {
  for (const total of [25.61, 40]) {
    const result = fit(
      { gpuBytes: 0, totalBytes: total * GB },
      {
        gpuCapacityGb: 0,
        totalCapacityGb: 32,
        systemRamCapacityGb: 32,
        freeGpuCapacityGb: 0,
        usableSystemRamGb: 24,
      },
    );
    assert.equal(result.cpuOnly, true);
    assert.match(result.advisory?.text ?? "", /RAM/);
    assert.doesNotMatch(result.advisory?.text ?? "", /CPU layers|GPU layers/);
  }
});

test("a confirmed zero free reading warns, while an unknown reading stays unknown", () => {
  for (const known of [false, true]) {
    const gpu = fit(
      { gpuBytes: 20 * GB, totalBytes: 25 * GB },
      { freeGpuCapacityGb: 0, freeGpuCapacityKnown: known },
    );
    assert.equal(gpu.freeGpuFit, known ? "exceeds" : "unknown");
    assert.equal(gpu.gpuFit, known ? "tight" : "fits");
    assert.equal(gpu.cpuOnly, false);
    const ram = fit(
      { gpuBytes: 0, totalBytes: 25 * GB },
      { usableSystemRamGb: 0, usableSystemRamKnown: known },
    );
    assert.equal(ram.hostPressured, known);
  }
});

test("shared pools keep their label and warn when known free memory reaches zero", () => {
  const result = fit(
    { gpuBytes: 0, totalBytes: 25 * GB },
    {
      freeGpuCapacityGb: 0,
      usableSystemRamGb: 0,
      freeGpuCapacityKnown: true,
      usableSystemRamKnown: true,
    },
    APPLE,
  );
  assert.equal(result.cpuOnly, false);
  assert.equal(result.hostPressured, true);
  assert.equal(result.gpuPressured, true);
});

test("compact estimates retain units and never round a lower bound up", () => {
  const candidates = memoryFigureCandidates(25.61 * GB, true);
  assert.equal(candidates[0], "≥ 25.61 GiB");
  assert.ok(candidates.includes("≥ 25.6 GiB"));
  assert.ok(candidates.includes("≥ 25 GiB"));
  assert.ok(!candidates.includes("≥ 26 GiB"));
  assert.ok(memoryFigureCandidates(2048 * GB, false).includes("2 TiB"));
});

test("the primary lower-bound label also rounds down", () => {
  assert.equal(memoryFigureCandidates(25.619 * GB, true)[0], "≥ 25.61 GiB");
  assert.equal(memoryFigureCandidates(25.619 * GB, false)[0], "25.62 GiB");
  for (const gib of [0.009, 25.619, 1024.999, 2048.129]) {
    for (const label of memoryFigureCandidates(gib * GB, true)) {
      const [, amount, unit] = label.split(" ");
      const scaled = Number(amount) * (unit === "TiB" ? 1024 : 1);
      assert.ok(scaled <= gib, `${label} exceeds ${gib} GiB`);
    }
  }
});

// ---------------------------------------------------------------------------
// The resident copy's own bytes, credited back to the free-memory questions.
//
// Studio unloads before it reloads, so those bytes return BEFORE the replacement allocates.
// Charging them counted a model against itself: opening Run settings for the model already
// loaded warned it would not fit the memory its own resident copy held.

test("reloading an unchanged config does not warn about the memory it is about to release", () => {
  // Apple 64 GB, a 25.61 GiB load, 25.53 GiB free because that same model is holding the rest.
  // Uncredited this is ratio 1.003 -> exceeds.
  const uncredited = fit(
    { gpuBytes: 25.61 * GB, totalBytes: 25.61 * GB },
    { freeGpuCapacityGb: 25.53, usableSystemRamGb: 25.53 },
    APPLE,
  );
  assert.equal(uncredited.usableHostFit, "exceeds");
  assert.ok(uncredited.advisory);

  const credited = fit(
    { gpuBytes: 25.61 * GB, totalBytes: 25.61 * GB },
    {
      freeGpuCapacityGb: 25.53,
      usableSystemRamGb: 25.53,
      reclaimableTotalBytes: 25.61 * GB,
      reclaimableGpuBytes: 25.61 * GB,
    },
    APPLE,
  );
  assert.equal(credited.gpuPressured, false);
  assert.equal(credited.hostPressured, false);
  assert.equal(credited.advisory, null);
});

test("growing the context past what the resident copy returns still warns", () => {
  // 40 GiB wanted, with 12 GiB free and 25 GiB returning on unload.
  const result = fit(
    { gpuBytes: 40 * GB, totalBytes: 40 * GB },
    {
      freeGpuCapacityGb: 12,
      usableSystemRamGb: 12,
      reclaimableTotalBytes: 25 * GB,
      reclaimableGpuBytes: 25 * GB,
    },
    APPLE,
  );
  assert.ok(result.gpuPressured || result.hostPressured);
  assert.equal(result.advisory?.text, ADVISORY_TEXTS.singlePoolPressure);
});

test("reclaimed memory preserves pressure thresholds in every pool", () => {
  for (const [wanted, expected] of [[34, "fits"], [35, "tight"], [40, "tight"], [41, "exceeds"]] as const) {
    const gpu = fit({ gpuBytes: wanted * GB, totalBytes: wanted * GB }, {
      freeGpuCapacityGb: 30,
      reclaimableTotalBytes: 10 * GB,
      reclaimableGpuBytes: 10 * GB,
    });
    assert.equal(gpu.freeGpuFit, expected);
    const host = fit({ gpuBytes: 0, totalBytes: wanted * GB }, {
      usableSystemRamGb: 30,
      reclaimableTotalBytes: 10 * GB,
    });
    assert.equal(host.usableHostFit, expected);
    const shared = fit({ gpuBytes: wanted * GB, totalBytes: wanted * GB }, {
      freeGpuCapacityGb: 30,
      usableSystemRamGb: 30,
      reclaimableTotalBytes: 10 * GB,
      reclaimableGpuBytes: 10 * GB,
    }, APPLE);
    assert.equal(shared.freeGpuFit, expected);
    assert.equal(shared.usableHostFit, expected);
  }
});

test("reclaimed memory does not turn missing free-memory readings into known ones", () => {
  for (const available of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
    const result = fit({ gpuBytes: 9 * GB, totalBytes: 9 * GB }, {
      freeGpuCapacityGb: available,
      usableSystemRamGb: available,
      reclaimableTotalBytes: 10 * GB,
      reclaimableGpuBytes: 10 * GB,
    }, APPLE);
    assert.equal(result.freeGpuFit, "unknown");
    assert.equal(result.usableHostFit, "unknown");
  }
  const exhausted = fit({ gpuBytes: 9 * GB, totalBytes: 9 * GB }, {
    freeGpuCapacityGb: 0,
    usableSystemRamGb: 0,
    freeGpuCapacityKnown: true,
    usableSystemRamKnown: true,
    reclaimableTotalBytes: 10 * GB,
    reclaimableGpuBytes: 10 * GB,
  }, APPLE);
  assert.equal(exhausted.freeGpuFit, "tight");
  assert.equal(exhausted.usableHostFit, "tight");
});

test("reclaimed memory first fills GPU and host reserve deficits", () => {
  const gpuDeficit = aggregateVramReserveDeficitGb([
    { memoryTotalGb: 24, memoryFreeGb: 0 },
  ], 0.9);
  assert.ok(Math.abs(gpuDeficit - 2.4) < 1e-9);
  const result = fit({ gpuBytes: 16 * GB, totalBytes: 32 * GB }, {
    freeGpuCapacityGb: 0,
    freeGpuCapacityKnown: true,
    freeGpuReserveDeficitGb: gpuDeficit,
    usableSystemRamGb: 0,
    usableSystemRamKnown: true,
    systemRamReserveDeficitGb: 2,
    reclaimableTotalBytes: 40 * GB,
    reclaimableGpuBytes: 20 * GB,
  });
  assert.equal(result.freeGpuFit, "tight");
  assert.equal(result.usableHostFit, "tight");
});

test("unmet reserves do not reduce memory already usable on another card", () => {
  for (const credit of [0, 1]) {
    const result = fit({ gpuBytes: 8 * GB, totalBytes: 8 * GB }, {
      freeGpuCapacityGb: 10,
      freeGpuReserveDeficitGb: 2.4,
      reclaimableTotalBytes: credit * GB,
      reclaimableGpuBytes: credit * GB,
    });
    assert.equal(result.freeGpuFit, "fits");
  }
  assert.equal(aggregateVramReserveDeficitGb([
    { memoryTotalGb: 24, memoryFreeGb: 5 },
  ], 0.9), 0);
});

test("the credit never improves a CAPACITY verdict, only a free-memory one", () => {
  // 70 GB on a 64 GB machine: unloading frees memory, it does not add any.
  const result = fit(
    { gpuBytes: 70 * GB, totalBytes: 70 * GB },
    {
      freeGpuCapacityGb: 2,
      usableSystemRamGb: 2,
      reclaimableTotalBytes: 70 * GB,
      reclaimableGpuBytes: 70 * GB,
    },
    APPLE,
  );
  assert.equal(result.totalFit, "exceeds");
  assert.equal(result.advisory?.tone, "warn");
  assert.equal(result.advisory?.text, ADVISORY_TEXTS.singlePoolExceeds);
});

test("on discrete memory the credit follows the pool each byte came from", () => {
  // 20 GB on the card, 20 GB on the host. The host credit is the part that was NOT on the GPU.
  const result = fit(
    { gpuBytes: 20 * GB, totalBytes: 40 * GB },
    {
      freeGpuCapacityGb: 4,
      usableSystemRamGb: 4,
      reclaimableTotalBytes: 40 * GB,
      reclaimableGpuBytes: 20 * GB,
    },
  );
  assert.equal(result.gpuPressured, false);
  assert.equal(result.hostPressured, false);
});

test("resident GPU credit requires the whole loaded pool to remain available", () => {
  const resident = { gpuBytes: 20 * GB, totalBytes: 40 * GB };
  for (const [loaded, requested, credited] of [
    [[0], [1], false],
    [[0, 1], [1], false],
    [[0], [0, 1], true],
    [[0, 1], [1, 0], true],
    [null, [0], false],
    [[0], null, true],
    [null, null, true],
  ] as const) {
    const credit = resolveReclaimableMemoryCredit(
      resident,
      { ids: loaded ? [...loaded] : null, indexKind: "physical" },
      { ids: requested ? [...requested] : null, indexKind: "physical" },
    );
    const result = fit(resident, {
      freeGpuCapacityGb: 4,
      usableSystemRamGb: 4,
      reclaimableTotalBytes: credit.totalBytes,
      reclaimableGpuBytes: credit.gpuBytes,
    });
    assert.equal(result.gpuPressured, !credited);
    assert.equal(result.hostPressured, false);
    assert.equal(credit.totalBytes - credit.gpuBytes, 20 * GB);
  }
});

test("resident GPU IDs from another namespace do not earn VRAM credit", () => {
  const credit = resolveReclaimableMemoryCredit(
    { gpuBytes: 20 * GB, totalBytes: 40 * GB },
    { ids: [0], indexKind: "physical" },
    { ids: [0], indexKind: "vulkan" },
  );
  assert.deepEqual(credit, { gpuBytes: 0, totalBytes: 20 * GB });
});

test("excluded VRAM credit never becomes extra RAM credit", () => {
  const credit = resolveReclaimableMemoryCredit(
    { gpuBytes: 20 * GB, totalBytes: 40 * GB },
    { ids: [0], indexKind: "physical" },
    { ids: [1], indexKind: "physical" },
  );
  const result = fit({ gpuBytes: 20 * GB, totalBytes: 50 * GB }, {
    freeGpuCapacityGb: 4,
    usableSystemRamGb: 4,
    reclaimableTotalBytes: credit.totalBytes,
    reclaimableGpuBytes: credit.gpuBytes,
  });
  assert.equal(result.gpuPressured, true);
  assert.equal(result.hostPressured, true);
});

test("CPU fallback and unmodelled experts withhold uncertain VRAM credit", () => {
  const pool = { ids: null, indexKind: null };
  for (const cpuFallback of [false, true]) {
    const credit = resolveReclaimableMemoryCredit(
      {
        gpuBytes: 20 * GB,
        totalBytes: 40 * GB,
        moeOffloadUnmodelled: !cpuFallback,
      },
      pool,
      pool,
      cpuFallback,
    );
    assert.deepEqual(credit, { gpuBytes: 0, totalBytes: 20 * GB });
    const result = fit({ gpuBytes: 20 * GB, totalBytes: 30 * GB }, {
      freeGpuCapacityGb: 4,
      usableSystemRamGb: 4,
      reclaimableTotalBytes: credit.totalBytes,
      reclaimableGpuBytes: credit.gpuBytes,
    });
    assert.equal(result.gpuPressured, true);
    assert.equal(result.hostPressured, false);
  }
});

test("shared GPU allocations return to RAM when the requested GPU changes", () => {
  const device: SystemGpuDevice = {
    index: 0,
    indexKind: "vulkan",
    name: "Shared GPU",
    memoryTotalGb: 32,
    memoryFreeGb: 0,
    sharedMemory: true,
    pinnable: true,
    diffusionPinnable: false,
  };
  const resident = { gpuBytes: 20 * GB, totalBytes: 24 * GB };
  const loaded = { ids: [0], indexKind: "vulkan" as const };
  const requested = { ids: [1], indexKind: "vulkan" as const };
  for (const topology of [device, { ...device, sharedMemory: false, unifiedMemory: true }]) {
    const credit = resolveReclaimableMemoryCredit(resident, loaded, requested, false, [topology]);
    assert.deepEqual(credit, { gpuBytes: 0, totalBytes: 24 * GB });
    const result = fit({ gpuBytes: 8 * GB, totalBytes: 28 * GB }, {
      usableSystemRamGb: 4,
      reclaimableTotalBytes: credit.totalBytes,
      reclaimableGpuBytes: credit.gpuBytes,
    });
    assert.equal(result.hostPressured, false);
  }
  const partial = resolveReclaimableMemoryCredit(resident, loaded, requested, false, [
    { ...device, sharedMemoryHostBackedGb: 24 },
  ]);
  assert.deepEqual(partial, { gpuBytes: 0, totalBytes: 16 * GB });
  const unknown = resolveReclaimableMemoryCredit(resident, loaded, requested, false, []);
  assert.deepEqual(unknown, { gpuBytes: 0, totalBytes: 4 * GB });
  const discrete = resolveReclaimableMemoryCredit(resident, loaded, requested, false, [
    { ...device, sharedMemory: false },
  ]);
  assert.deepEqual(discrete, unknown);
});

test("capacity advice does not assume a pageable load mode", () => {
  for (const [gpu, total] of [[0, 100], [30, 100], [8, 78]]) {
    const result = fit({ gpuBytes: gpu * GB, totalBytes: total * GB }, {});
    assert.match(result.advisory?.text ?? "", /smaller model/);
    assert.doesNotMatch(result.advisory?.text ?? "", /paging|will load|will fit/);
  }
});

test("a GPU credit larger than its own total cannot inflate the host share", () => {
  // GPU share above its own total. Clamping keeps the host credit at zero instead of negative.
  const result = fit(
    { gpuBytes: 20 * GB, totalBytes: 40 * GB },
    {
      freeGpuCapacityGb: 23,
      usableSystemRamGb: 60,
      reclaimableTotalBytes: 10 * GB,
      reclaimableGpuBytes: 999 * GB,
    },
  );
  // 20 GB host share less a credit floored at zero, against 60 GB usable.
  assert.equal(result.hostPressured, false);
  assert.ok(Number.isFinite(result.hostShareBytes));
});

test("a garbage credit cannot increase available memory", () => {
  for (const credit of [
    Number.NaN,
    Number.POSITIVE_INFINITY,
    -50 * GB,
    undefined,
  ]) {
    const result = fit(
      { gpuBytes: 30 * GB, totalBytes: 30 * GB },
      {
        freeGpuCapacityGb: 6,
        usableSystemRamGb: 6,
        reclaimableTotalBytes: credit,
        reclaimableGpuBytes: credit,
      },
      APPLE,
    );
    // The warning a real 30 GB load under 6 GB free deserves, unchanged.
    assert.ok(
      result.gpuPressured || result.hostPressured,
      `credit ${String(credit)} should not have silenced the warning`,
    );
  }
});

test("an unmeasurable footprint stays unknown with reclaimed capacity", () => {
  const result = fit(
    { gpuBytes: Number.NaN, totalBytes: Number.NaN },
    {
      freeGpuCapacityGb: 6,
      usableSystemRamGb: 6,
      reclaimableTotalBytes: 30 * GB,
      reclaimableGpuBytes: 30 * GB,
    },
    APPLE,
  );
  assert.equal(result.freeGpuFit, "unknown");
  assert.equal(result.usableHostFit, "unknown");
});
