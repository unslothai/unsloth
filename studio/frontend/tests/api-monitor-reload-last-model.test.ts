// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  RELOAD_MISSING_HISTORY_MESSAGE,
  reloadLastModel,
  resolveReloadTarget,
  type ReloadTarget,
} from "../src/features/api-monitor/reload-last-model.ts";

test("reload target uses the durable last local model, not the selected checkpoint", () => {
  const resolved = resolveReloadTarget(
    { id: "unsloth/qwen3", kind: "model", ggufVariant: null },
    "external::openai:gpt-5",
    (id) => id.startsWith("external::"),
    (id, variant) => ({
      config: { id, variant, maxSeqLength: 8192 },
    }),
  );

  assert.equal(resolved.ok, true);
  if (!resolved.ok) return;
  assert.deepEqual(resolved.target, {
    id: "unsloth/qwen3",
    kind: "model",
    ggufVariant: null,
    config: { id: "unsloth/qwen3", variant: null, maxSeqLength: 8192 },
    forceReload: true,
    externalPreserved: true,
  });
});

test("reload target keeps a GGUF quant for cached repos", () => {
  const resolved = resolveReloadTarget(
    { id: "unsloth/qwen3-GGUF", kind: "gguf", ggufVariant: "Q4_K_M" },
    "",
    () => false,
    (id, variant) => ({ config: { key: `${id}:${variant}` } }),
  );

  assert.equal(resolved.ok, true);
  if (!resolved.ok) return;
  assert.equal(resolved.target.kind, "gguf");
  assert.equal(resolved.target.ggufVariant, "Q4_K_M");
  assert.deepEqual(resolved.target.config, {
    key: "unsloth/qwen3-GGUF:Q4_K_M",
  });
});

test("reload rejects a quant-less cached GGUF repo", () => {
  const resolved = resolveReloadTarget(
    { id: "unsloth/qwen3-GGUF", kind: "gguf", ggufVariant: null },
    "",
    () => false,
    () => null,
  );

  assert.deepEqual(resolved, {
    ok: false,
    reason: RELOAD_MISSING_HISTORY_MESSAGE,
  });
});

test("reloadLastModel awaits durable history before loading", async () => {
  const loaded: ReloadTarget[] = [];
  const target = await reloadLastModel({
    readLastLoad: async () => ({
      id: "unsloth/llama",
      kind: "model",
      ggufVariant: null,
    }),
    readSelectedCheckpoint: () => "",
    isExternalSelection: () => false,
    resolveConfig: () => null,
    loadTarget: async (next) => {
      loaded.push(next);
    },
  });

  assert.equal(target.id, "unsloth/llama");
  assert.equal(target.forceReload, true);
  assert.deepEqual(loaded, [target]);
});
