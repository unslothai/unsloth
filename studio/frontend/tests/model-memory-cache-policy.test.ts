// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The estimate cache is what stops a long model list costing one metadata read
// per row, so its key has to be exactly as specific as the request. Two rows
// that would ask the backend different questions must not share an answer, and
// a row that could not be sized must not stay blank for the rest of the session.

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { estimateCacheKey, estimateIsUnsized } = await import(
  "../src/lib/model-memory.ts"
);

const BASE = { repoId: "unsloth/gemma-4-12b-it-GGUF", quant: "Q4_K_M" };

test("an omitted slot count is not the same request as an explicit one slot", () => {
  // The request omits n_parallel when there is no override, and the backend
  // fills in the server's standing count, which defaults above one. Keying both
  // as 1 served a single-slot answer to a default-slot row.
  assert.notEqual(
    estimateCacheKey({ ...BASE }),
    estimateCacheKey({ ...BASE, nParallel: 1 }),
  );
});

test("a non-positive slot count reads as the server default", () => {
  const serverDefault = estimateCacheKey({ ...BASE });
  assert.equal(estimateCacheKey({ ...BASE, nParallel: 0 }), serverDefault);
  assert.equal(estimateCacheKey({ ...BASE, nParallel: null }), serverDefault);
  assert.equal(
    estimateCacheKey({ ...BASE, nParallel: undefined }),
    serverDefault,
  );
});

test("distinct slot counts key apart", () => {
  assert.notEqual(
    estimateCacheKey({ ...BASE, nParallel: 1 }),
    estimateCacheKey({ ...BASE, nParallel: 4 }),
  );
});

test("every input that changes the answer changes the key", () => {
  const base = estimateCacheKey(BASE);
  const variants = [
    { ...BASE, sizeBytes: 1 },
    { ...BASE, nCtx: 4096 },
    { ...BASE, kvCacheDtype: "q8_0" },
    { ...BASE, speculativeType: "mtp" },
    { ...BASE, nParallel: 2 },
    { ...BASE, quant: "Q5_K_M" },
    { ...BASE, repoId: "unsloth/other-GGUF" },
  ];
  for (const v of variants) {
    assert.notEqual(estimateCacheKey(v), base);
  }
  // and they are all distinct from one another
  const keys = variants.map(estimateCacheKey);
  assert.equal(new Set(keys).size, keys.length);
});

test("a re-download under a stable quant name re-keys", () => {
  // Same repo and quant, different file: the cached weights would otherwise
  // outrank the row's fresh size.
  assert.notEqual(
    estimateCacheKey({ ...BASE, sizeBytes: 7_000_000_000 }),
    estimateCacheKey({ ...BASE, sizeBytes: 7_100_000_000 }),
  );
});

test("a native-context row is distinct from one pinned to a number", () => {
  assert.notEqual(
    estimateCacheKey({ ...BASE }),
    estimateCacheKey({ ...BASE, nCtx: 131072 }),
  );
});

test("the key is stable for the same inputs", () => {
  assert.equal(
    estimateCacheKey({ ...BASE, nCtx: 4096, nParallel: 4 }),
    estimateCacheKey({ ...BASE, nCtx: 4096, nParallel: 4 }),
  );
});

test("a 200 that sized nothing counts as unsized", () => {
  assert.equal(
    estimateIsUnsized({ kvBytes: null, weightsBytes: null, specBytes: null }),
    true,
  );
});

test("any figure at all means the answer is real", () => {
  // Weights alone is a real answer: a model whose header cannot be read still
  // charts its file size, and that must not expire in 30 seconds.
  assert.equal(
    estimateIsUnsized({
      kvBytes: null,
      weightsBytes: 7 * 1024 ** 3,
      specBytes: null,
    }),
    false,
  );
  assert.equal(
    estimateIsUnsized({ kvBytes: 1, weightsBytes: null, specBytes: null }),
    false,
  );
  assert.equal(
    estimateIsUnsized({ kvBytes: null, weightsBytes: null, specBytes: 1 }),
    false,
  );
});

test("a zero figure is a measurement, not a missing one", () => {
  assert.equal(
    estimateIsUnsized({ kvBytes: 0, weightsBytes: 0, specBytes: 0 }),
    false,
  );
});

// --- settings that must reach the request, and the ones that must suppress it ---

const { extraArgsOwnPlacement, PLACEMENT_OWNING_ARGS } = await import(
  "../src/lib/model-memory.ts"
);

test("draft depth and checkpoints key apart, including zero", () => {
  // Zero is a real choice for both (no rollback states, no checkpoints), so it
  // must not collapse into "unset" the way a `?? ""` would make it.
  for (const field of ["specDraftNMax", "ctxCheckpoints"] as const) {
    const unset = estimateCacheKey({ ...BASE });
    const zero = estimateCacheKey({ ...BASE, [field]: 0 });
    const some = estimateCacheKey({ ...BASE, [field]: 16 });
    assert.notEqual(zero, unset, `${field}: 0 collapsed into unset`);
    assert.notEqual(some, zero);
    assert.notEqual(some, unset);
  }
});

test("draft cache dtype and vision both re-key", () => {
  assert.notEqual(
    estimateCacheKey({ ...BASE, specDraftCacheType: "q8_0" }),
    estimateCacheKey({ ...BASE }),
  );
  assert.notEqual(
    estimateCacheKey({ ...BASE, disableVision: true }),
    estimateCacheKey({ ...BASE }),
  );
});

test("pass-through args that own placement make the bar abstain", () => {
  // These are appended after Unsloth's own flags, so they decide where the load
  // runs and the VRAM total stops describing it.
  for (const flag of PLACEMENT_OWNING_ARGS) {
    assert.equal(
      extraArgsOwnPlacement([flag, "0"]),
      true,
      `${flag} did not suppress the bar`,
    );
    // The "--flag=value" argv shape has to be recognised too.
    assert.equal(extraArgsOwnPlacement([`${flag}=0`]), true, `${flag}=0`);
  }
});

test("ordinary pass-through args do not suppress the bar", () => {
  // Sampling and logging flags say nothing about placement.
  assert.equal(extraArgsOwnPlacement(null), false);
  assert.equal(extraArgsOwnPlacement(undefined), false);
  assert.equal(extraArgsOwnPlacement([]), false);
  assert.equal(extraArgsOwnPlacement(["--temp", "0.7", "--verbose"]), false);
  // A near-miss must not match on a prefix. `--device-draft` is not a spelling
  // any parser accepts, so it stays out.
  assert.equal(extraArgsOwnPlacement(["--device-draft"]), false);
  // `--gpu-layers-draft` used to be listed here as a near-miss, which was wrong:
  // it is in the backend's _DRAFT_GPU_LAYER_FLAGS and does own placement, just
  // the drafter's rather than the target's. A drafter pinned off the GPU is host
  // memory the bar would otherwise charge to the card.
  assert.equal(extraArgsOwnPlacement(["--gpu-layers-draft"]), true);
});
