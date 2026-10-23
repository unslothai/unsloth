// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

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

const IDLE_DISCRETE: MemoryFitCapacity = {
  gpuCapacityGb: 24,
  totalCapacityGb: 88,
  systemRamCapacityGb: 64,
  freeGpuCapacityGb: 23,
  usableSystemRamGb: 60,
  singleMemoryPool: false,
};

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

test("a footprint well inside the capacity fits", () => {
  assert.equal(classifyMemoryFit(8 * GB, 24), "fits");
});

test("above 85% of the capacity is tight, above 100% exceeds", () => {
  assert.equal(classifyMemoryFit(20.5 * GB, 24), "tight");
  assert.equal(classifyMemoryFit(25 * GB, 24), "exceeds");
  assert.equal(classifyMemoryFit(0.85 * 24 * GB, 24), "fits");
  assert.equal(classifyMemoryFit(24 * GB, 24), "tight");
});

test("nothing probed and nothing to weigh are both no verdict", () => {
  assert.equal(classifyMemoryFit(8 * GB, 0), "unknown");
  assert.equal(classifyMemoryFit(0, 24), "unknown");
  assert.equal(classifyMemoryFit(8 * GB, -4), "unknown");
  assert.equal(classifyMemoryFit(-8 * GB, 24), "unknown");
});

// NaN and Infinity fail every comparison, so they used to fall through to fits.
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
    "Fits system RAM, but little is free right now. Free memory, or try a shorter context or smaller model.",
  gpuPressure:
    "Fits this GPU, but little VRAM is free right now. Free memory or try Auto context.",
};

test("D1: a single-pool host under memory pressure now says so", () => {
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
  const gpuSide = fit(
    { gpuBytes: 30 * GB, totalBytes: 30 * GB },
    { freeGpuCapacityGb: 6, usableSystemRamGb: 60 },
    APPLE,
  );
  assert.equal(gpuSide.advisory?.text, ADVISORY_TEXTS.singlePoolPressure);
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

test("pressure advice does not shift layers into another pressured pool", () => {
  for (const freeGpu of [24, 22, 0]) {
    const result = fit(
      { gpuBytes: 22 * GB, totalBytes: 40 * GB },
      {
        freeGpuCapacityGb: freeGpu,
        usableSystemRamGb: 10,
        reclaimableTotalBytes: 10 * GB,
      },
    );
    assert.equal(result.rawGpuFit, "tight");
    assert.equal(result.usableHostFit, "tight");
    assert.doesNotMatch(result.advisory?.text ?? "", /CPU layers|offload/);
    assert.match(result.advisory?.text ?? "", /shorter context or smaller model/);
  }
  for (const [gpuGb, hostGb, freeGpu, freeHost] of [
    [30, 5, 23, 2],
    [21, 65, 1, 60],
  ]) {
    const result = fit(
      { gpuBytes: gpuGb * GB, totalBytes: (gpuGb + hostGb) * GB },
      { freeGpuCapacityGb: freeGpu, usableSystemRamGb: freeHost },
    );
    assert.equal(result.advisory?.tone, "warn");
    assert.doesNotMatch(result.advisory?.text ?? "", /layers|offload/);
    assert.match(result.advisory?.text ?? "", /shorter context or smaller model/);
  }
});

test("a discrete host under VRAM pressure alone gets the card wording", () => {
  const result = fit(
    { gpuBytes: 20 * GB, totalBytes: 20 * GB },
    { freeGpuCapacityGb: 8 },
  );
  assert.equal(result.advisory?.text, ADVISORY_TEXTS.gpuPressure);
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

test("one pool weighs the WHOLE load against what is free, not the GPU share", () => {
  // On one pool, CPU-offloaded bytes come from the same memory.
  const pooled = fit(
    { gpuBytes: 6 * GB, totalBytes: 30 * GB },
    { freeGpuCapacityGb: 10, usableSystemRamGb: 10 },
    { ...APPLE, singleMemoryPool: true },
  );
  assert.equal(pooled.freeGpuFit, "exceeds");
  assert.equal(pooled.gpuPressured, true);
  const discrete = fit(
    { gpuBytes: 6 * GB, totalBytes: 30 * GB },
    { freeGpuCapacityGb: 10 },
  );
  assert.equal(discrete.freeGpuFit, "fits");
});

test("the host share is the bytes outside the GPU, floored at zero", () => {
  assert.equal(fit({ gpuBytes: 10 * GB, totalBytes: 30 * GB }, {}).hostShareBytes, 20 * GB);
  assert.equal(fit({ gpuBytes: 30 * GB, totalBytes: 10 * GB }, {}).hostShareBytes, 0);
});

test("one pool asks no separate host-share question", () => {
  assert.equal(fit({ gpuBytes: 6 * GB, totalBytes: 30 * GB }, {}, APPLE).hostShareFit, "unknown");
});

test("a non-finite footprint produces no verdict and no advisory, and does not throw", () => {
  for (const bad of [Number.NaN, Number.POSITIVE_INFINITY]) {
    const result = fit({ gpuBytes: bad, totalBytes: bad }, {});
    assert.equal(result.gpuFit, "unknown");
    assert.equal(result.totalFit, "unknown");
    assert.equal(result.advisory, null);
    // Math.max(0, NaN) is NaN, not 0.
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
      // a drives both gpuBytes and gpuCapacityGb; b only reaches the free reading.
      if (!Number.isFinite(a)) {
        assert.notEqual(result.gpuFit, "fits", `gpuBytes=${String(a)}`);
      }
      if (!Number.isFinite(b)) {
        assert.notEqual(result.totalFit, "fits", `totalBytes=${String(b)}`);
      }
      assert.ok(
        Number.isFinite(result.hostShareBytes) && result.hostShareBytes >= 0,
        `hostShareBytes=${result.hostShareBytes}`,
      );
    }
  }
});

test("a figure is always finite and never negative", () => {
  // GiB, not GB: the divide is by 1024**3.
  assert.equal(formatMemoryGb(24 * GB), "24.00 GiB");
  assert.equal(formatMemoryGb(0), "0.00 GiB");
  assert.equal(formatMemoryGb(-5 * GB), "0.00 GiB");
  assert.equal(formatMemoryGb(Number.NaN), "0.00 GiB");
  assert.equal(formatMemoryGb(Number.POSITIVE_INFINITY), "0.00 GiB");
  assert.equal(formatMemoryGb(undefined as unknown as number), "0.00 GiB");
});

test("the KV caption names the dtype, what was priced, and where it lives", () => {
  assert.equal(
    resolveKvNote({ cacheTypeKv: "q8_0", nCtx: 32768, nParallel: 1, kvOnGpu: true }),
    "q8_0 · 32,768 tokens",
  );
  assert.equal(
    resolveKvNote({ cacheTypeKv: null, nCtx: 4096, nParallel: 4, kvOnGpu: false }),
    "f16 · 4,096 tokens · 4 slots · host RAM",
  );
  assert.equal(
    resolveKvNote({ cacheTypeKv: "f16", nCtx: 4096, nParallel: 1, kvOnGpu: false }),
    "f16 · 4,096 tokens · host RAM",
  );
});

test("the KV caption survives a field that is not a number", () => {
  // toLocaleString() on null throws.
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
  // --spec-draft-ngl 0 moves the drafter; --no-kv-offload moves the target.
  assert.equal(resolveDraftCacheNote(0, 4 * GB), "host RAM");
  assert.equal(resolveDraftCacheNote(1 * GB, 4 * GB), "1.00 GiB on GPU");
  assert.equal(resolveDraftCacheNote(4 * GB, 4 * GB), undefined);
  assert.equal(resolveDraftCacheNote(Number.NaN, 4 * GB), "host RAM");
});

test("an unsizable pass-through adapter marks the total a floor", () => {
  // Unstatable --lora / --control-vector files make the figure a lower bound.
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

test("capacity advice does not assume a pageable load mode", () => {
  for (const [gpu, total] of [[0, 100], [30, 100], [8, 78]]) {
    const result = fit({ gpuBytes: gpu * GB, totalBytes: total * GB }, {});
    assert.match(result.advisory?.text ?? "", /smaller model/);
    assert.doesNotMatch(result.advisory?.text ?? "", /paging|will load|will fit/);
  }
});

const EIGHT_GB_CARD: Partial<MemoryFitCapacity> = {
  gpuCapacityGb: 7.2,
  totalCapacityGb: 39.2,
  systemRamCapacityGb: 32,
  freeGpuCapacityGb: 7,
  usableSystemRamGb: 22,
};
const NATIVE_QWEN3_8B = { gpuBytes: 10.45 * GB, totalBytes: 10.45 * GB, nCtx: 40960 };
const QWEN3_8B_FLOOR = 4.82 * GB;
const floorAt = (bytes: number) => ({
  contextIsPinned: false,
  gpuFloorBytes: bytes,
  floorCanOffload: true,
});

test("an unpinned context over the card is not an overage when the floor fits", () => {
  const result = fit(
    { ...NATIVE_QWEN3_8B, ...floorAt(QWEN3_8B_FLOOR) },
    EIGHT_GB_CARD,
  );
  assert.equal(result.gpuFit, "fits");
  assert.equal(result.totalFit, "fits");
  assert.equal(
    result.advisory?.text,
    "Estimated at the full 40,960-token context. Auto context will shrink it to fit.",
  );
  assert.equal(result.advisory?.tone, "muted");
});

test("a pinned context over the card still exceeds, as does an older backend's answer", () => {
  for (const pin of [{ contextIsPinned: true }, {}]) {
    const result = fit({ ...NATIVE_QWEN3_8B, ...pin }, EIGHT_GB_CARD);
    assert.equal(result.gpuFit, "exceeds", JSON.stringify(pin));
    assert.equal(result.advisory?.text, ADVISORY_TEXTS.gpuExceeds);
  }
});

test("an unpinned context without a floor keeps the native verdict", () => {
  const result = fit(
    { ...NATIVE_QWEN3_8B, contextIsPinned: false, gpuFloorBytes: null },
    EIGHT_GB_CARD,
  );
  assert.equal(result.gpuFit, "exceeds");
});

test("a floor over the card is a real overage, and the note does not send the user to Auto", () => {
  const result = fit(
    {
      gpuBytes: 14 * GB,
      totalBytes: 14 * GB,
      nCtx: 131072,
      ...floorAt(9 * GB),
    },
    EIGHT_GB_CARD,
  );
  assert.equal(result.gpuFit, "exceeds");
  assert.equal(result.advisory?.tone, "warn");
  assert.equal(
    result.advisory?.text,
    "Exceeds GPU memory even at the shortest context the loader tries, so some layers will run on the CPU and generation will be slower.",
  );
});

test("an unpinned context that already fits prints no note", () => {
  const result = fit(
    { gpuBytes: 4 * GB, totalBytes: 4 * GB, nCtx: 8192, ...floorAt(3.5 * GB) },
    EIGHT_GB_CARD,
  );
  assert.equal(result.gpuFit, "fits");
  assert.equal(result.advisory, null);
});

test("a single pool draws the same verdict from the floor", () => {
  const result = fit(
    { gpuBytes: 70 * GB, totalBytes: 70 * GB, nCtx: 262144, ...floorAt(20 * GB) },
    {},
    APPLE,
  );
  assert.equal(result.totalFit, "fits");
  assert.equal(
    result.advisory?.text,
    "Estimated at the full 262,144-token context. Auto context will shrink it to fit.",
  );
});

test("free-memory pressure outranks the Auto note, since the loader fits against what is free", () => {
  const result = fit(
    { ...NATIVE_QWEN3_8B, ...floorAt(5.92 * GB) },
    { ...EIGHT_GB_CARD, freeGpuCapacityGb: 3 },
  );
  assert.equal(result.gpuFit, "tight");
  assert.equal(
    result.advisory?.text,
    "Fits this GPU, but little VRAM is free right now, so Auto may pick a shorter context or run some layers on the CPU. Free memory first.",
  );
});

test("on one pool, pressure outranks the Auto note too", () => {
  const result = fit(
    { gpuBytes: 70 * GB, totalBytes: 70 * GB, nCtx: 262144, ...floorAt(20 * GB) },
    { freeGpuCapacityGb: 10, usableSystemRamGb: 10 },
    APPLE,
  );
  assert.equal(
    result.advisory?.text,
    "Fits this machine, but little memory is free right now, so Auto may pick a shorter context. Free memory first.",
  );
});

test("pressure still outranks the Auto note when the floor itself is tight", () => {
  // A 7.05 GiB floor on a 7.2 GiB budget classifies tight rather than fits.
  const result = fit(
    { ...NATIVE_QWEN3_8B, ...floorAt(7.05 * GB) },
    { ...EIGHT_GB_CARD, freeGpuCapacityGb: 3 },
  );
  assert.equal(result.rawGpuFit, "tight");
  assert.equal(
    result.advisory?.text,
    "Fits this GPU, but little VRAM is free right now, so Auto may pick a shorter context or run some layers on the CPU. Free memory first.",
  );
});

test("a pinned tight fit under pressure gets the pressure note too", () => {
  const result = fit({ gpuBytes: 7.05 * GB, totalBytes: 7.05 * GB }, { ...EIGHT_GB_CARD, freeGpuCapacityGb: 3 });
  assert.equal(result.rawGpuFit, "tight");
  assert.equal(
    result.advisory?.text,
    "Fits this GPU, but little VRAM is free right now. Free memory or try Auto context.",
  );
});

test("a floor over the card says loading may fail where layers cannot move", () => {
  const result = fit(
    { gpuBytes: 14 * GB, totalBytes: 14 * GB, nCtx: 131072, ...floorAt(9 * GB), floorCanOffload: false },
    EIGHT_GB_CARD,
  );
  assert.equal(result.gpuFit, "exceeds");
  assert.equal(
    result.advisory?.text,
    "Exceeds GPU memory even at the shortest context the loader tries, and these settings keep layers from moving to the CPU, so loading may fail.",
  );
});

test("pressure where layers cannot move promises only a shorter context", () => {
  const result = fit(
    { ...NATIVE_QWEN3_8B, ...floorAt(5.92 * GB), floorCanOffload: false },
    { ...EIGHT_GB_CARD, freeGpuCapacityGb: 3 },
  );
  assert.equal(
    result.advisory?.text,
    "Fits this GPU, but little VRAM is free right now, so Auto may pick a shorter context. Free memory first.",
  );
});

const SMALL_HOST = { systemRamCapacityGb: 16, totalCapacityGb: 23.2, usableSystemRamGb: 15 };

test("a context whose GPU share fits is opened native, so its host share is too", () => {
  const result = fit(
    { gpuBytes: 2 * GB, totalBytes: 30 * GB, nCtx: 262144, ...floorAt(1 * GB) },
    { ...EIGHT_GB_CARD, ...SMALL_HOST },
  );
  assert.equal(result.hostShareBytes, 28 * GB);
  assert.equal(result.hostShareFit, "exceeds");
});

test("a shrunk context keeps its host terms at native, the bound that holds", () => {
  const result = fit(
    { gpuBytes: 9 * GB, totalBytes: 29 * GB, nCtx: 131072, ...floorAt(5 * GB) },
    { ...EIGHT_GB_CARD, ...SMALL_HOST },
  );
  assert.equal(result.gpuFit, "fits");
  assert.equal(result.hostShareBytes, 20 * GB);
  assert.equal(result.hostShareFit, "exceeds");
});
