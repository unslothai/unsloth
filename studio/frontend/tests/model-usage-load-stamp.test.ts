// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// "Recent" orders by load, so only a selection that is actually loaded is stamped. A remembered
// model shown in the trigger without being loaded must leave the load times alone.

import assert from "node:assert/strict";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

const { store } = installLocalStorageFake();
const { loadStampId, loadedAt, recordModelLoaded } = await import(
  "../src/features/model-picker/components/model-selector/model-usage.ts"
);

const KEY = "unsloth.model-load-times.v1";

test("an unloaded selection is not stamped", () => {
  assert.equal(loadStampId("unsloth/Kokoro-82M-GGUF", false), undefined);
  assert.equal(loadStampId(undefined, true), undefined);
  assert.equal(loadStampId(null), undefined);
  assert.equal(loadStampId("", true), undefined);
  assert.equal(store.has(KEY), false);
});

test("a loaded selection, or one with no loaded signal, still stamps", () => {
  assert.equal(loadStampId("unsloth/Kokoro-82M-GGUF", true), "unsloth/Kokoro-82M-GGUF");
  assert.equal(loadStampId("unsloth/Qwen3-8B"), "unsloth/Qwen3-8B");
  const id = loadStampId("Unsloth/Qwen3-8B", true);
  assert.ok(id);
  const times = recordModelLoaded(id);
  assert.ok(loadedAt(times, "unsloth/qwen3-8b") > 0);
  assert.ok(store.get(KEY)?.includes("unsloth/qwen3-8b"));
});
