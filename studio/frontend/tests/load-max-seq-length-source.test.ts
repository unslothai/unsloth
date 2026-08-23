// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Reloading the same GGUF replays the context OUR fitter resolved last time, so
// the sheet keeps showing what the user sees. That replay is also what makes the
// value look user-supplied to the backend, which is how a forced drafter ended up
// inheriting a context fit while the drafter was dropped (#9550). The value sent
// is unchanged; only its provenance is now reported, so the wrapper must keep
// returning exactly what it did before.

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { resolveLoadMaxSeqLength, resolveLoadMaxSeqLengthDetailed } =
  await import("../src/features/chat/presets/preset-policy.ts");

const GGUF = {
  modelId: "unsloth/Qwen3-8B-GGUF",
  ggufVariant: "Q4_K_M",
  isGguf: true,
  customContextLength: null as number | null,
  ggufContextLength: null as number | null,
  currentCheckpoint: "",
  activeGgufVariant: null as string | null,
  maxSeqLength: 4096,
  presetSource: "user" as const,
};

test("a same-model GGUF reload is marked as our own resolved context", () => {
  const resolved = resolveLoadMaxSeqLengthDetailed({
    ...GGUF,
    ggufContextLength: 112896,
    currentCheckpoint: GGUF.modelId,
    activeGgufVariant: "Q4_K_M",
  });
  assert.equal(resolved.value, 112896);
  assert.equal(resolved.source, "resident-reload");
});

test("a context the user pinned is not ours to re-fit", () => {
  const resolved = resolveLoadMaxSeqLengthDetailed({
    ...GGUF,
    customContextLength: 65536,
    ggufContextLength: 112896,
    currentCheckpoint: GGUF.modelId,
    activeGgufVariant: "Q4_K_M",
  });
  assert.equal(resolved.value, 65536);
  assert.equal(resolved.source, "user-pinned");
});

test("a first load asks the fitter and is not a replay", () => {
  const resolved = resolveLoadMaxSeqLengthDetailed(GGUF);
  assert.equal(resolved.value, 0);
  assert.equal(resolved.source, "gguf-auto");
});

test("a builtin-default preset asks the fitter too", () => {
  const resolved = resolveLoadMaxSeqLengthDetailed({
    ...GGUF,
    presetSource: "builtin-default",
    ggufContextLength: 112896,
    currentCheckpoint: GGUF.modelId,
    activeGgufVariant: "Q4_K_M",
  });
  assert.equal(resolved.value, 0);
  assert.equal(resolved.source, "builtin-default");
});

test("a transformers load carries its own maxSeqLength", () => {
  const resolved = resolveLoadMaxSeqLengthDetailed({
    ...GGUF,
    isGguf: false,
    ggufVariant: null,
    modelId: "unsloth/Qwen3-8B",
  });
  assert.equal(resolved.value, 4096);
  assert.equal(resolved.source, "transformers");
});

test("the wrapper returns exactly the detailed value in every branch", () => {
  const cases = [
    GGUF,
    { ...GGUF, customContextLength: 65536 },
    {
      ...GGUF,
      ggufContextLength: 112896,
      currentCheckpoint: GGUF.modelId,
      activeGgufVariant: "Q4_K_M",
    },
    { ...GGUF, presetSource: "builtin-default" as const },
    { ...GGUF, isGguf: false, ggufVariant: null, modelId: "unsloth/Qwen3-8B" },
  ];
  for (const args of cases) {
    assert.equal(
      resolveLoadMaxSeqLength(args),
      resolveLoadMaxSeqLengthDetailed(args).value,
    );
  }
});
