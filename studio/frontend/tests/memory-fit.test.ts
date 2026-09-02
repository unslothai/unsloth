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

import {
  type MemoryFitCapacity,
  type MemoryFitEstimate,
  type MemoryFitVerdict,
  classifyMemoryFit,
  formatMemoryGb,
  resolveDraftCacheNote,
  resolveKvNote,
  resolveMemoryFit,
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
    "More than this machine's memory. The GPU and the rest of the system share one pool here, so there is nothing to offload to.",
  singlePoolPressure:
    "This fits the machine, but not what is free right now. If that memory is not the model being replaced, the context will be fitted down or the load refused.",
  hostShareExceeds:
    "More than system RAM holds. This placement keeps most of the load outside the GPU, and spare VRAM cannot take those bytes.",
  totalExceeds:
    "More than this machine holds. The GPU and system RAM together are not enough for this load, so spilling layers or fitting the context down will not recover it.",
  gpuExceeds:
    "More than this GPU holds. Layers will spill to system RAM, or the context will be fitted down to what fits.",
  hostPressure:
    "The part of this load that runs from system RAM fits the machine, but not what is free right now. If that memory is not the model being replaced, the load will be refused.",
  gpuPressure:
    "This fits the card, but something is using it right now. If that memory is not the model being replaced, layers will spill or the context will be fitted down.",
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
  assert.match(drafter.advisory?.text ?? "", /fetch rather than one on this disk/);
  const moe = fit({ moeOffloadUnmodelled: true, totalBytes: 900 * GB }, {});
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
  assert.equal(result.advisory?.text, ADVISORY_TEXTS.hostShareExceeds);
});

test("a load beyond GPU and RAM combined, with a host share that fits RAM", () => {
  const result = fit(
    { gpuBytes: 80 * GB, totalBytes: 120 * GB },
    { gpuCapacityGb: 24, totalCapacityGb: 88, systemRamCapacityGb: 64 },
  );
  assert.equal(result.advisory?.text, ADVISORY_TEXTS.totalExceeds);
});

test("a load that only overflows the card is told it will spill", () => {
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
  const sized = resolveMemoryFit(
    { ...SIZED, totalBytes: 8 * GB, gpuBytes: 8 * GB },
    IDLE_DISCRETE,
  );
  assert.equal(sized.bounded, false);
});
