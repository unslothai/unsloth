// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { ResidentRunTarget } from "../src/features/api-monitor/slots-settings-request.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  residentHasSlotsSetting,
  residentMatchesMonitor,
  slotsSettingsRequest,
} = await import("../src/features/api-monitor/slots-settings-request.ts");

const GGUF: ResidentRunTarget = {
  checkpoint: "unsloth/Qwen3-4B-GGUF",
  ggufVariant: "UD-Q4_K_XL",
  isGguf: true,
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

test("models without a Parallel Slots field stay a plain readout", () => {
  assert.equal(residentHasSlotsSetting(null), false);
  assert.equal(slotsSettingsRequest(null, "req"), null);
  // Transformers, MLX and NPU loads: the monitor reports no llama-server slots for them.
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

test("the readout is offered only when the picker takes the target for the monitor's model", () => {
  const monitor = { id: "unsloth/Qwen3-4B-GGUF", ggufVariant: "UD-Q4_K_XL" };
  assert.equal(residentMatchesMonitor(GGUF, monitor), true);
  // An API auto-switch to another quant of the same repo, before Chat's runtime catches up:
  // Run would reload the old quant.
  assert.equal(
    residentMatchesMonitor(GGUF, { ...monitor, ggufVariant: "Q8_0" }),
    false,
  );
  // A file-picked GGUF: Chat holds the path, the monitor its bare stem. The picker cannot tie
  // them, so Run settings would open on stored settings rather than the loaded ones.
  const picked = {
    ...GGUF,
    checkpoint: "/home/user/models/qwen3-4b-Q4_K_M.gguf",
    ggufVariant: "Q4_K_M",
    nativePathToken: "tok",
    nativePathExpiresAtMs: Number.MAX_SAFE_INTEGER,
  };
  assert.equal(
    residentMatchesMonitor(picked, {
      id: "qwen3-4b-Q4_K_M",
      ggufVariant: "Q4_K_M",
    }),
    false,
  );
  assert.equal(residentMatchesMonitor(null, monitor), false);
});
