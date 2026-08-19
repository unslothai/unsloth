// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What the requested MLX speculation tuple is allowed to be, wherever it is stored.
// The pin and the method are one setting: a drafter pinned for MTP that outlived a
// switch back to Auto would override the pick Auto exists to make.

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  registerStoreStubResolver,
} from "./helpers/kit.ts";

registerStoreStubResolver();
installLocalStorageFake();

const {
  mlxSpeculativeLoadFields,
  normalizeMlxDraftBlockSize,
  normalizeMlxDraftModel,
  normalizeMlxSpeculativeMode,
} = await import("../src/lib/speculative-modes.ts");
const { toApiOverride } = await import(
  "../src/features/model-picker/api/model-overrides.ts"
);
const { DEFAULT_PER_MODEL_CONFIG } = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);

test("an absent mode takes the caller's default, an unusable one is refused", () => {
  assert.equal(normalizeMlxSpeculativeMode(undefined, "auto"), "auto");
  assert.equal(normalizeMlxSpeculativeMode(" MTP ", "auto"), "mtp");
  assert.equal(normalizeMlxSpeculativeMode("ngram", "auto"), "off");
});

test("a pinned drafter survives only under the method that pinned it", () => {
  for (const method of ["mtp", "dflash", "eagle3"] as const) {
    assert.equal(normalizeMlxDraftModel(" org/d ", method), "org/d");
  }
  assert.equal(normalizeMlxDraftModel("org/d", "auto"), null);
  assert.equal(normalizeMlxDraftModel("org/d", "off"), null);
});

test("a draft block size is clamped to what the backend accepts, and dropped when off", () => {
  assert.equal(normalizeMlxDraftBlockSize(1, "auto"), 2);
  assert.equal(normalizeMlxDraftBlockSize(99, "auto"), 16);
  assert.equal(normalizeMlxDraftBlockSize(4.4, "mtp"), 4);
  assert.equal(normalizeMlxDraftBlockSize(8, "off"), null);
});

test("a server override travels only for a pin the server can act on", () => {
  const override = (config: Record<string, unknown>) =>
    toApiOverride({ ...DEFAULT_PER_MODEL_CONFIG, ...config });
  assert.deepEqual(
    override({
      mlxSpeculativeMode: "mtp",
      mlxDraftModel: "org/d",
      mlxDraftBlockSize: 4,
    }),
    {
      mlx_speculative_mode: "mtp",
      mlx_draft_model: "org/d",
      mlx_draft_block_size: 4,
    },
  );
  // Auto re-resolves per load and Off is the server default, so neither is sent.
  assert.deepEqual(
    override({ mlxSpeculativeMode: "auto", mlxDraftModel: "org/d" }),
    {},
  );
  assert.deepEqual(
    override({ mlxSpeculativeMode: "off", mlxDraftModel: "org/d" }),
    {},
  );
  assert.deepEqual(override({ mlxSpeculativeMode: "mtp" }), {});
});

test("a load sends the tuple only for a model MLX will serve", () => {
  const pinned = {
    mlxSpeculativeMode: "mtp",
    mlxDraftModel: "org/d",
    mlxDraftBlockSize: 6,
  };
  assert.deepEqual(mlxSpeculativeLoadFields(pinned, true), {
    mlx_speculative_mode: "mtp",
    mlx_draft_model: "org/d",
    mlx_draft_block_size: 6,
  });
  assert.deepEqual(mlxSpeculativeLoadFields(pinned, false), {
    mlx_speculative_mode: "off",
    mlx_draft_model: null,
    mlx_draft_block_size: null,
  });
  // A model with nothing remembered asks for Auto, not for Off: Off is a choice.
  assert.equal(mlxSpeculativeLoadFields({}, true).mlx_speculative_mode, "auto");
});
