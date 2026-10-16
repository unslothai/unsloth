// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { hfModelFitsDevice } from "../src/features/model-picker/components/model-selector/recommended-fit.ts";
import { readSrcAsync } from "./helpers/kit.ts";

const [chatTab, store, pickers, hubPage, settingsSearch] = await Promise.all([
  readSrcAsync("features/settings/tabs/chat-tab.tsx"),
  readSrcAsync("features/chat/stores/chat-runtime-store.ts"),
  readSrcAsync("features/model-picker/components/model-selector/pickers.tsx"),
  readSrcAsync("features/hub/hub-page.tsx"),
  readSrcAsync("features/settings/settings-search.ts"),
]);

const GB = 1024 ** 3;

test("Settings exposes the same fit toggle the picker and Hub read", () => {
  assert.match(
    chatTab,
    /useChatRuntimeStore\(\(state\) => state\.fitOnDeviceOnly\)/,
  );
  assert.match(
    chatTab,
    /checked=\{fitOnDeviceOnly\}\s*onCheckedChange=\{setFitOnDeviceOnly\}/,
  );
  assert.match(
    chatTab,
    /aria-label=\{t\("settings\.chat\.modelSelection\.fitOnDeviceOnly"\)\}/,
  );
  assert.ok(
    settingsSearch.includes('"settings.chat.modelSelection.fitOnDeviceOnly"'),
  );
});

test("the fit filter stays opt-in", () => {
  assert.match(
    store,
    /fitOnDeviceOnly: loadBool\(MODELS_FIT_ON_DEVICE_ONLY_KEY, false\)/,
  );
});

test("with the setting off no list pays for a fit check", () => {
  // Every gate tests the flag first, so `||` never reaches the fit predicate when it is off.
  const gates = [...pickers.matchAll(/!fitOnDeviceOnly \|\|/g)].length;
  assert.ok(gates >= 2, `picker search gates: ${gates}`);
  assert.match(pickers, /!deviceFiltered \|\| fits\(r\)/);
  const hubGates = [...hubPage.matchAll(/!fitOnDeviceOnly \|\|/g)].length;
  assert.ok(hubGates >= 2, `hub gates: ${hubGates}`);
});

test("a model is hidden only when it exceeds GPU memory plus RAM", () => {
  const gpu = { memoryTotalGb: 8, systemRamAvailableGb: 32, budgetKnown: true };
  assert.equal(
    hfModelFitsDevice(
      { id: "unsloth/a-GGUF", isGguf: true, estimatedSizeBytes: 4 * GB },
      gpu,
    ),
    true,
  );
  assert.equal(
    hfModelFitsDevice(
      { id: "unsloth/b-GGUF", isGguf: true, estimatedSizeBytes: 14 * GB },
      gpu,
    ),
    true,
  );
  assert.equal(
    hfModelFitsDevice(
      { id: "unsloth/c-GGUF", isGguf: true, estimatedSizeBytes: 40 * GB },
      gpu,
    ),
    false,
  );
  assert.equal(
    hfModelFitsDevice(
      { id: "unsloth/c-GGUF", isGguf: true, estimatedSizeBytes: 40 * GB },
      { memoryTotalGb: 0, systemRamAvailableGb: 0 },
    ),
    true,
  );
});

test("a safetensors row on a discrete GPU is sized like its badge, with RAM only where layers swap", () => {
  // 24 GB of VRAM beside 64 GB of RAM.
  const cuda = { memoryTotalGb: 24, systemRamAvailableGb: 64, budgetKnown: true, backend: "cuda" };
  const xpu = { ...cuda, backend: "xpu" };
  const row = (id: string, totalParams: number) => ({ id, totalParams, isGguf: false });
  // Badge estimates (0.9 B/param + 1.4 GB): 24B 23 GB (tight), 32B 30.2, 70B 64.4, 120B 109.4.
  // Several cards, or XPU, cannot offload: VRAM alone, kept unless the badge says exceeds.
  assert.equal(hfModelFitsDevice(row("org/m-24B", 24e9), cuda, { gpuCount: 2 }), true);
  assert.equal(hfModelFitsDevice(row("org/m-32B", 32e9), cuda, { gpuCount: 2 }), false);
  assert.equal(hfModelFitsDevice(row("org/m-32B", 32e9), xpu), false);
  // One CUDA or ROCm card block-swaps layers into host RAM.
  assert.equal(hfModelFitsDevice(row("org/m-70B", 70e9), cuda, { gpuCount: 1 }), true);
  assert.equal(hfModelFitsDevice(row("org/m-70B", 70e9), { ...cuda, backend: "rocm" }), true);
  assert.equal(hfModelFitsDevice(row("org/m-120B", 120e9), cuda, { gpuCount: 1 }), false);
  // A prequantized checkpoint loads as stored (23e9 B is 21.4 GiB), where only bnb block-swaps, or
  // rebuilt as bnb 4-bit, which block-swaps too.
  const prequant = (quantMethod: string, bytes: number) => ({
    ...row("org/m-32B", 32e9),
    quantMethod,
    estimatedSizeBytes: bytes,
  });
  assert.equal(hfModelFitsDevice(prequant("mxfp4", 23e9), xpu), true);
  assert.equal(hfModelFitsDevice(prequant("mxfp4", 40e9), cuda), false);
  assert.equal(hfModelFitsDevice(prequant("bitsandbytes", 40e9), cuda), true);
  assert.equal(hfModelFitsDevice(prequant("compressed-tensors", 100e9), cuda), true);
  assert.equal(hfModelFitsDevice(prequant("fp8", 100e9), cuda), false);
  // With no byte size, a packed checkpoint is sized as the smallest quant, not the bnb rebuild.
  assert.equal(hfModelFitsDevice({ ...row("org/m-32B", 32e9), quantMethod: "gptq" }, xpu), true);
  // Host pools, MLX ids, Images / Video rows and GGUF keep the quant rule.
  assert.equal(hfModelFitsDevice(row("org/m-32B", 32e9), xpu, { hostPooledMemory: true }), true);
  assert.equal(hfModelFitsDevice(row("mlx-community/m-32B", 32e9), xpu), true);
  assert.equal(hfModelFitsDevice(row("org/m-32B", 32e9), xpu, { mediaLoad: true }), true);
  assert.equal(
    hfModelFitsDevice({ id: "org/m-32B-GGUF", totalParams: 32e9, isGguf: true }, xpu),
    true,
  );
});
