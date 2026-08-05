// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The cache reconciliation effect re-points the training selection at whatever
// copy of the repo the inventory reports. A repo can legitimately exist under
// more than one HF cache root, so "first usable row wins" would silently move an
// explicit selection to a different copy on the next inventory tick. These are
// behavioural: they call the selector and assert which path comes back.

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { selectModelCacheReference } = await import(
  "../src/features/training/lib/model-cache-reference.ts"
);

const MODEL = "unsloth/Llama-3.2-1B-Instruct";
const PRIMARY = "/cache-a/models--unsloth--Llama-3.2-1B-Instruct";
const SECONDARY = "/cache-b/models--unsloth--Llama-3.2-1B-Instruct";

function cached(cachePath: string, overrides: Record<string, unknown> = {}) {
  return {
    repo_id: MODEL,
    cache_path: cachePath,
    model_format: "safetensors",
    partial: false,
    load_id: null,
    ...overrides,
    // biome-ignore lint/suspicious/noExplicitAny: inventory row shape is wider than this fixture needs
  } as any;
}

function local(path: string, overrides: Record<string, unknown> = {}) {
  return {
    id: MODEL,
    path,
    source: "hf_cache",
    model_format: "safetensors",
    partial: false,
    load_id: null,
    ...overrides,
    // biome-ignore lint/suspicious/noExplicitAny: inventory row shape is wider than this fixture needs
  } as any;
}

test("a selected cache path is kept when the inventory lists another copy first", () => {
  const reference = selectModelCacheReference(
    [cached(SECONDARY), cached(PRIMARY)],
    [],
    MODEL,
    PRIMARY,
  );

  assert.deepEqual(reference, {
    localPath: PRIMARY,
    modelFormat: "safetensors",
  });
});

test("with no selection the first usable copy still wins", () => {
  const reference = selectModelCacheReference(
    [cached(SECONDARY), cached(PRIMARY)],
    [],
    MODEL,
    null,
  );

  assert.equal(reference?.localPath, SECONDARY);
});

test("a selected copy that left the inventory is promoted to a remaining one", () => {
  const reference = selectModelCacheReference(
    [cached(SECONDARY)],
    [],
    MODEL,
    PRIMARY,
  );

  assert.equal(reference?.localPath, SECONDARY);
});

test("a selected copy that is no longer usable does not pin the selection", () => {
  // Partial and non-trainable rows are filtered before preference is applied, so
  // an exact path match on an unusable row must not beat a usable sibling.
  for (const broken of [
    cached(PRIMARY, { partial: true }),
    cached(PRIMARY, { model_format: "gguf" }),
  ]) {
    const reference = selectModelCacheReference(
      [broken, cached(SECONDARY)],
      [],
      MODEL,
      PRIMARY,
    );
    assert.equal(reference?.localPath, SECONDARY);
  }
});

test("the local hf_cache fallback honours the selected path too", () => {
  const reference = selectModelCacheReference(
    [],
    [local(SECONDARY), local(PRIMARY)],
    MODEL,
    PRIMARY,
  );

  assert.equal(reference?.localPath, PRIMARY);
});

test("path preference tolerates separator and case differences", () => {
  const reference = selectModelCacheReference(
    [cached(SECONDARY), cached(`${PRIMARY}/`)],
    [],
    MODEL,
    PRIMARY,
  );

  assert.equal(reference?.localPath, `${PRIMARY}/`);
});

test("a repo absent from both inventories resolves to nothing", () => {
  assert.equal(
    selectModelCacheReference([], [], MODEL, PRIMARY),
    null,
  );
  assert.equal(
    selectModelCacheReference([cached(PRIMARY, { repo_id: "other/model" })], [], MODEL, null),
    null,
  );
});

test("non hf_cache local rows are never selected", () => {
  assert.equal(
    selectModelCacheReference([], [local(PRIMARY, { source: "ollama" })], MODEL, PRIMARY),
    null,
  );
});
