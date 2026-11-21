// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  RELOAD_MISSING_HISTORY_MESSAGE,
  type ReloadTarget,
  reloadLastModel,
  resolveReloadTarget,
} from "../src/features/api-monitor/reload-last-model.ts";

test("reload target carries the remembered config", () => {
  assert.deepEqual(
    resolveReloadTarget(
      { id: "unsloth/qwen3", kind: "model", ggufVariant: null },
      (id, variant) => ({ config: { id, variant, maxSeqLength: 8192 } }),
    ),
    {
      id: "unsloth/qwen3",
      kind: "model",
      ggufVariant: null,
      config: { id: "unsloth/qwen3", variant: null, maxSeqLength: 8192 },
    },
  );
});

test("reload target keeps a GGUF quant for cached repos", () => {
  const target = resolveReloadTarget(
    { id: "unsloth/qwen3-GGUF", kind: "gguf", ggufVariant: "Q4_K_M" },
    (id, variant) => ({ config: { key: `${id}:${variant}` } }),
  );
  assert.equal(target?.ggufVariant, "Q4_K_M");
  assert.deepEqual(target?.config, { key: "unsloth/qwen3-GGUF:Q4_K_M" });
});

test("reload target accepts a direct .gguf file without a quant", () => {
  const target = resolveReloadTarget(
    { id: "/models/qwen3-Q4_K_M.gguf", kind: "gguf", ggufVariant: null },
    () => null,
  );
  assert.deepEqual(target, {
    id: "/models/qwen3-Q4_K_M.gguf",
    kind: "gguf",
    ggufVariant: null,
    config: null,
  });
});

test("reloadLastModel rejects missing history without loading", async () => {
  const loaded: ReloadTarget[] = [];
  await assert.rejects(
    reloadLastModel({
      readLastLoad: async () => null,
      resolveConfig: () => null,
      loadTarget: async (next) => {
        loaded.push(next);
      },
    }),
    { message: RELOAD_MISSING_HISTORY_MESSAGE },
  );
  assert.deepEqual(loaded, []);
});

test("reloadLastModel awaits durable history before loading", async () => {
  const loaded: ReloadTarget[] = [];
  const target = await reloadLastModel({
    readLastLoad: async () => ({
      id: "unsloth/llama",
      kind: "model",
      ggufVariant: null,
    }),
    resolveConfig: () => null,
    loadTarget: async (next) => {
      loaded.push(next);
    },
  });
  assert.equal(target.id, "unsloth/llama");
  assert.deepEqual(loaded, [target]);
});
