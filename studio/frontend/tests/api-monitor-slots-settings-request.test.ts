// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { ResidentRunTarget } from "../src/features/api-monitor/slots-settings-request.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  keptRunSettingsRequest,
  residentHasRunSettings,
  residentHasSlotsSetting,
  residentMatchesMonitor,
  runSettingsRequest,
  slotsSettingsRequest,
} = await import("../src/features/api-monitor/slots-settings-request.ts");
const { modelConfigTarget } = await import(
  "../src/features/model-picker/model-config/model-config-handoff.ts"
);

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

test("Settings opens Advanced without focusing Parallel Slots or changing the load identity", () => {
  const resident = {
    ...GGUF,
    loadId: "/cache/qwen",
    nativePathToken: "token",
    nativePathExpiresAtMs: 1_000,
  };
  const request = runSettingsRequest(resident, "settings", 999);
  const slots = slotsSettingsRequest(resident, "slots", 999);
  assert.equal(request?.focus, "advanced");
  assert.equal(request?.id, resident.checkpoint);
  assert.equal(request?.meta.loadId, resident.loadId);
  assert.equal(request?.meta.nativePathToken, "token");
  assert.deepEqual(request?.meta, slots?.meta);
  assert.equal(runSettingsRequest(resident, "expired", 1_000), null);
  assert.equal(residentHasRunSettings(null), false);
});

test("Settings also supports non-GGUF residents without offering a Slots shortcut", () => {
  const resident = {
    ...GGUF,
    checkpoint: "unsloth/Qwen3-0.6B",
    isGguf: false,
    ggufVariant: null,
    isLora: true,
  };
  assert.equal(residentHasRunSettings(resident), true);
  assert.equal(
    residentMatchesMonitor(resident, {
      id: resident.checkpoint,
      ggufVariant: null,
    }),
    true,
  );
  assert.equal(runSettingsRequest(resident, "settings")?.meta.isLora, true);
  assert.equal(slotsSettingsRequest(resident, "slots"), null);
});

test("a GGUF resident opens its Run settings on Parallel Slots, as Chat's reload names it", () => {
  assert.deepEqual(slotsSettingsRequest(GGUF, "req-1"), {
    requestId: "req-1",
    id: "unsloth/Qwen3-4B-GGUF",
    focus: "parallelSlots",
    standalone: true,
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

test("a kept model's settings open on its own checkpoint and quant", () => {
  const gguf = keptRunSettingsRequest(
    { checkpoint: "unsloth/gemma-3-270m-it-GGUF", quant: "Q4_K_M" },
    "kept",
  );
  assert.equal(gguf.id, "unsloth/gemma-3-270m-it-GGUF");
  assert.equal(gguf.focus, "advanced");
  assert.equal(gguf.meta.ggufVariant, "Q4_K_M");
  assert.equal(gguf.meta.isGguf, true);
  assert.equal(gguf.standalone, true);
  assert.equal(runSettingsRequest(GGUF, "live")?.standalone, true);
  const safetensors = keptRunSettingsRequest(
    { checkpoint: "unsloth/Qwen3-0.6B", quant: null },
    "kept",
  );
  assert.equal(safetensors.meta.ggufVariant, undefined);
  assert.equal(safetensors.meta.isGguf, false);
});

test("a kept standalone GGUF opens its saved settings, keyed without the filename quant", () => {
  const request = keptRunSettingsRequest(
    { checkpoint: "/models/gemma-3-270m-it-Q4_K_M.gguf", quant: "Q4_K_M" },
    "kept",
  );
  assert.equal(request.meta.ggufVariant, undefined);
  assert.equal(request.meta.isGguf, true);
  assert.equal(
    keptRunSettingsRequest(
      { checkpoint: "unsloth/gemma-3-270m-it-GGUF", quant: "Q4_K_M" },
      "kept",
    ).meta.ggufVariant,
    "Q4_K_M",
  );
});

test("a kept HF cache snapshot opens the repo's saved settings and still loads the snapshot", () => {
  const snapshot =
    "/Users/me/.cache/huggingface/hub/models--unsloth--Qwen3-0.6B/snapshots/abc123";
  const request = keptRunSettingsRequest({ checkpoint: snapshot, quant: null }, "kept");
  const target = modelConfigTarget(request.id, request.meta);
  assert.equal(target.configId, "unsloth/Qwen3-0.6B");
  assert.equal(target.id, snapshot);
});
