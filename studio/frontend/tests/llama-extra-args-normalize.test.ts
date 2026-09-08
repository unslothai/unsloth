// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The three pass-through resolutions `/load` performs before its already-loaded
 * comparator runs, mirrored so the resident-model shortcut judges the request the server
 * would actually receive rather than the one the panel typed.
 */

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const {
  parseGpuLayersOverride,
  resolveTensorParallel,
  stripManagedOffloadFlags,
} = await import("../src/features/chat/lib/llama-extra-args-normalize.ts");

test("an explicit split mode last-wins over the toggle", () => {
  assert.equal(resolveTensorParallel(null, true), true);
  assert.equal(resolveTensorParallel([], false), false);
  assert.equal(resolveTensorParallel(["--split-mode", "tensor"], false), true);
  assert.equal(resolveTensorParallel(["--split-mode", "layer"], true), false);
  assert.equal(resolveTensorParallel(["-sm", "row"], true), false);
  // Both spellings, and the last one wins.
  assert.equal(
    resolveTensorParallel(["-sm=tensor", "--split-mode", "none"], true),
    false,
  );
  assert.equal(
    resolveTensorParallel(["--split-mode", "none", "-sm=TENSOR"], false),
    true,
  );
  // A flag with no value says nothing about what was requested.
  assert.equal(resolveTensorParallel(["--split-mode"], true), true);
  assert.equal(
    resolveTensorParallel(["--split-mode", "--flash-attn"], true),
    true,
  );
});

test("a pass-through layer count is read the way the route reads it", () => {
  const absent = { kind: "absent" };
  const invalid = { kind: "invalid" };
  const value = (layers: number) => ({ kind: "value", layers });

  assert.deepEqual(parseGpuLayersOverride(null), absent);
  assert.deepEqual(parseGpuLayersOverride(["--flash-attn", "on"]), absent);
  assert.deepEqual(parseGpuLayersOverride(["-ngl", "20"]), value(20));
  assert.deepEqual(parseGpuLayersOverride(["--gpu-layers=0"]), value(0));
  assert.deepEqual(parseGpuLayersOverride(["--n-gpu-layers", "-1"]), value(-1));
  assert.deepEqual(
    parseGpuLayersOverride(["-ngl", "8", "-ngl", "99"]),
    value(99),
  );
  // Below -1 and non-integers are what the backend rejects outright, and it RAISES
  // rather than defaulting, so these stay distinct from an absent override.
  assert.deepEqual(parseGpuLayersOverride(["-ngl", "-2"]), invalid);
  assert.deepEqual(parseGpuLayersOverride(["-ngl", "many"]), invalid);
  assert.deepEqual(parseGpuLayersOverride(["-ngl", "20.5"]), invalid);
  // -1 is a value, not a flag: shorts always start with a letter. Without that rule the
  // commonest pass-through of all, an explicit Auto, reads as malformed.
  assert.deepEqual(
    parseGpuLayersOverride(["-ngl", "-1", "--flash-attn", "on"]),
    value(-1),
  );
  // llama.cpp takes the underscore spelling of a long option, and so does this.
  assert.deepEqual(parseGpuLayersOverride(["--n_gpu_layers", "12"]), value(12));
});

test("the offload family is dropped with its values, and nothing else is", () => {
  assert.deepEqual(stripManagedOffloadFlags(null), null);
  assert.deepEqual(stripManagedOffloadFlags(undefined), undefined);
  assert.deepEqual(stripManagedOffloadFlags([]), []);
  assert.deepEqual(
    stripManagedOffloadFlags(["-ngl", "20", "--flash-attn", "on"]),
    ["--flash-attn", "on"],
  );
  assert.deepEqual(stripManagedOffloadFlags(["--gpu-layers=20"]), []);
  assert.deepEqual(
    stripManagedOffloadFlags(["--fit", "on", "-ncmoe", "4", "--cpu-moe"]),
    [],
  );
  // A split mode is not offload, and manual does not own it.
  assert.deepEqual(stripManagedOffloadFlags(["-sm", "tensor"]), [
    "-sm",
    "tensor",
  ]);
  // A trailing offload flag with no value takes nothing with it.
  assert.deepEqual(stripManagedOffloadFlags(["--flash-attn", "on", "-ngl"]), [
    "--flash-attn",
    "on",
  ]);
});
