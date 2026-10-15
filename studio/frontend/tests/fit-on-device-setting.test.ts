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

test("a safetensors row on a discrete GPU is kept only when its badge would not say exceeds", () => {
  // 24 GB card with 64 GB of RAM, which a transformers load cannot offload into.
  const gpu = { memoryTotalGb: 24, systemRamAvailableGb: 64, budgetKnown: true };
  const row = (id: string, totalParams: number) => ({ id, totalParams, isGguf: false });
  // 14B: 14 * 0.9 + 1.4 = 14 GB, under the card.
  assert.equal(hfModelFitsDevice(row("org/m-14B", 14e9), gpu), true);
  // 24B: 23 GB, the badge's "tight" band, still kept.
  assert.equal(hfModelFitsDevice(row("org/m-24B", 24e9), gpu), true);
  // 32B and 70B: 30.2 and 64.4 GB, badged "exceeds".
  assert.equal(hfModelFitsDevice(row("org/m-32B", 32e9), gpu), false);
  assert.equal(hfModelFitsDevice(row("org/m-70B", 70e9), gpu), false);
  // A host memory pool and MLX ids keep the quant rule.
  assert.equal(
    hfModelFitsDevice(row("org/m-32B", 32e9), gpu, { hostPooledMemory: true }),
    true,
  );
  assert.equal(hfModelFitsDevice(row("mlx-community/m-32B", 32e9), gpu), true);
  // GGUF rows are unchanged: the same 32B repo as a GGUF is kept on its quant size.
  assert.equal(
    hfModelFitsDevice({ id: "org/m-32B-GGUF", totalParams: 32e9, isGguf: true }, gpu),
    true,
  );
});
