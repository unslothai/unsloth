// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import assert from "node:assert/strict";
import test from "node:test";
import {
  confirmManagedEngineIfNeeded,
  offeredEngines,
  useManagedEngineOfferStore,
} from "../src/features/model-picker/hooks/managed-engine-offer.ts";
import { readText } from "./helpers/kit.ts";

// #11728: a Default-engine load of a compressed-tensors / AWQ / GPTQ checkpoint pauses on this offer.
const OFFER = { quantization: "compressed-tensors", engines: ["vllm" as const] };
const store = () => useManagedEngineOfferStore.getState();

test("no offer, or an offer naming no engine this build knows, never opens the dialog", async () => {
  assert.equal(await confirmManagedEngineIfNeeded("m", null), null);
  assert.equal(await confirmManagedEngineIfNeeded("m", { quantization: "awq", engines: [] }), null);
  const unknown = { quantization: "awq", engines: ["tgi"] } as unknown as typeof OFFER;
  assert.equal(await confirmManagedEngineIfNeeded("m", unknown), null);
  assert.equal(store().open, false);
  assert.deepEqual(offeredEngines({ quantization: "gptq", engines: ["sglang", "vllm"] }), ["sglang", "vllm"]);
});

test("choosing the engine resumes the load with it; declining resumes with null", async () => {
  const chosen = confirmManagedEngineIfNeeded("unsloth/Qwen3.8-27B-NVFP4", OFFER);
  assert.equal(store().open, true);
  assert.equal(store().modelName, "unsloth/Qwen3.8-27B-NVFP4");
  assert.deepEqual(store().offer, OFFER);
  store().resolve("vllm");
  assert.equal(await chosen, "vllm");
  assert.equal(store().open, false);

  const declined = confirmManagedEngineIfNeeded("m", OFFER);
  store().resolve(null);
  assert.equal(await declined, null);
  store().resolve("vllm");
});

test("a newer load declines the older one, so one click never resumes two loads", async () => {
  const first = confirmManagedEngineIfNeeded("first", OFFER);
  const second = confirmManagedEngineIfNeeded("second", OFFER);
  assert.equal(await first, null);
  assert.equal(store().modelName, "second");
  store().resolve("vllm");
  assert.equal(await second, "vllm");
});

test("a cancelled load closes its dialog and an already-cancelled one never opens it", async () => {
  const controller = new AbortController();
  const pending = confirmManagedEngineIfNeeded("m", OFFER, controller.signal);
  assert.equal(store().open, true);
  controller.abort();
  assert.equal(await pending, null);
  assert.equal(store().open, false);
  assert.equal(await confirmManagedEngineIfNeeded("m", OFFER, controller.signal), null);
  assert.equal(store().open, false);
  const later = new AbortController();
  const chosen = confirmManagedEngineIfNeeded("m", OFFER, later.signal);
  store().resolve("vllm");
  later.abort();
  assert.equal(await chosen, "vllm");
});

test("the chat load asks before unloading anything and loads with the chosen engine", () => {
  const source = readText("../src/features/chat/hooks/use-chat-model-runtime.ts");
  const offerAt = source.indexOf("confirmManagedEngineIfNeeded(");
  const unloadAt = source.indexOf("await unloadModel({ model_path: currentCheckpoint })");
  assert.ok(offerAt > 0 && unloadAt > offerAt, "the offer must come before the preliminary unload");
  assert.match(source, /loadModel\(\{\s*model_path: loadPath,\s*\.\.\.loadEngineFields,/);
  const revalidateAt = source.indexOf("...loadEngineFields,", offerAt);
  assert.ok(revalidateAt > offerAt && revalidateAt < unloadAt, "the chosen engine must be validated before the unload");
  // Keep loaded models: the switch re-asks about running chats and then replaces, like any managed load.
  const branch = source.slice(offerAt, unloadAt);
  assert.match(branch, /if \(keepsOthers\) \{[\s\S]*?stopDecision = await confirmStopRunningChatsIfNeeded\([\s\S]*?keepsOthers = false;\s*forceCancelActive = stopDecision\.forceCancelActive;\s*loadRun\.forceCancelActive = forceCancelActive;/);
  assert.match(readText("../src/app/routes/__root.tsx"), /<ManagedEngineOfferDialog \/>/);
});

test("an install started in the dialog stays mounted until it hands the load over", () => {
  // Unmounting on the first ready poll would drop EngineInstall's success effect, and with it onUse.
  const dialog = readText("../src/features/model-picker/components/managed-engine-offer-dialog.tsx");
  assert.match(dialog, /ready\.length === 0 \|\| installing\.includes\(engine\.engine\)/);
  assert.match(dialog, /engine\.job\.state === "running"/);
  assert.match(dialog, /onUse=\{\(\) => resolve\(engine\.engine\)\}/);
});
