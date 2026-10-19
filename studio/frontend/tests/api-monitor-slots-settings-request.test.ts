// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  type ResidentRunTarget,
  residentHasSlotsSetting,
  slotsSettingsRequest,
} from "../src/features/api-monitor/slots-settings-request.ts";

const GGUF: ResidentRunTarget = {
  checkpoint: "unsloth/Qwen3-4B-GGUF",
  ggufVariant: "UD-Q4_K_XL",
  isGguf: true,
  isMlx: false,
  isLora: false,
  isDiffusion: false,
  loadId: null,
  nativePathToken: null,
  nativePathExpiresAtMs: null,
};

test("a GGUF resident opens its Run settings on Parallel Slots, as Chat's reload names it", () => {
  assert.deepEqual(slotsSettingsRequest(GGUF, "req-1"), {
    requestId: "req-1",
    id: "unsloth/Qwen3-4B-GGUF",
    focus: "parallelSlots",
    meta: {
      source: "local",
      isLora: false,
      loadId: null,
      ggufVariant: "UD-Q4_K_XL",
      nativePathToken: undefined,
      nativePathExpiresAtMs: null,
      isGguf: true,
      isDiffusion: false,
      isDownloaded: true,
    },
  });
});

test("an MLX resident has the field too", () => {
  const mlx = {
    ...GGUF,
    checkpoint: "mlx-community/Qwen3.5-2B-4bit",
    ggufVariant: null,
    isGguf: false,
    isMlx: true,
  };
  assert.equal(residentHasSlotsSetting(mlx), true);
  assert.equal(slotsSettingsRequest(mlx, "req-2")?.meta.ggufVariant, undefined);
});

test("models without a Parallel Slots field stay a plain readout", () => {
  assert.equal(residentHasSlotsSetting(null), false);
  assert.equal(slotsSettingsRequest(null, "req"), null);
  // Transformers / NPU loads have no slots setting.
  assert.equal(residentHasSlotsSetting({ ...GGUF, isGguf: false }), false);
  // The diffusion runner never receives --parallel.
  assert.equal(residentHasSlotsSetting({ ...GGUF, isDiffusion: true }), false);
});

test("a file-picked GGUF whose access expired cannot be reloaded, so it is not offered", () => {
  const picked = {
    ...GGUF,
    nativePathToken: "tok",
    nativePathExpiresAtMs: 1_000,
  };
  assert.equal(residentHasSlotsSetting(picked, 999), true);
  assert.equal(
    slotsSettingsRequest(picked, "req", 999)?.meta.nativePathToken,
    "tok",
  );
  assert.equal(residentHasSlotsSetting(picked, 1_000), false);
});
