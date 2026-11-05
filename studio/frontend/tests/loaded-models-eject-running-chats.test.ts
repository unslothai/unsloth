// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Ejecting a model with a running chat must send a forced unload, not surface the 409.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

register("./helpers/loaded-models-eject-resolver.mjs", import.meta.url);

const stub = await import("./helpers/loaded-models-eject-stub.ts");
const { ejectLoadedModel } = await import(
  "../src/features/loaded-models/loaded-models-api.ts"
);

const QWEN = "unsloth/Qwen3-4B-GGUF";
const ORNITH = "ornith-ai/Ornith-1.5-35B-A3B-GGUF";

function chatRow(name: string, inactive = false) {
  return {
    id: `chat:${name}`,
    kind: "text" as const,
    source: "chat" as const,
    name,
    detail: inactive ? "Still in memory" : "GGUF",
    ...(inactive ? { inactive: true } : {}),
  };
}

test("a generating model is ejected once stopping its chat is confirmed", async () => {
  stub.resetStub({ active: QWEN, loaded: [QWEN], generating: true });
  stub.state.confirmAnswer = { proceed: true, forceCancelActive: true };
  const outcome = await ejectLoadedModel(chatRow(QWEN));
  assert.deepEqual(stub.state.confirmCalls, [
    { action: "Unloading this model", effect: "unloadOne", model: QWEN },
  ]);
  assert.equal(stub.state.unloads[0].force_cancel_active, true);
  assert.deepEqual(stub.state.queueStops, [["running-thread"]]);
  assert.deepEqual(outcome, { status: "ejected" });
  assert.equal(stub.state.active, null);
});

test("a row replaced since the poll stops no queue and unloads nothing", async () => {
  stub.resetStub({ active: ORNITH, loaded: [ORNITH], generating: true });
  stub.state.confirmAnswer = { proceed: true, forceCancelActive: true };
  const outcome = await ejectLoadedModel(chatRow(QWEN));
  assert.deepEqual(stub.state.unloads, []);
  assert.deepEqual(stub.state.queueStops, []);
  assert.deepEqual(outcome, { status: "replaced", resident: ORNITH });
});

test("a generating model kept in memory beside the active one is ejected too", async () => {
  stub.resetStub({ active: QWEN, loaded: [QWEN, ORNITH], generating: true });
  stub.state.confirmAnswer = { proceed: true, forceCancelActive: true };
  const outcome = await ejectLoadedModel(chatRow(ORNITH, true));
  assert.deepEqual(stub.state.unloads, [
    { model_path: ORNITH, force_cancel_active: true },
  ]);
  assert.deepEqual(outcome, { status: "ejected" });
  assert.deepEqual(stub.state.loaded, [QWEN]);
});

test("a cached row another tab already evicted stops no queue", async () => {
  stub.resetStub({ active: QWEN, loaded: [QWEN], generating: true });
  stub.state.confirmAnswer = { proceed: true, forceCancelActive: true };
  const outcome = await ejectLoadedModel(chatRow(ORNITH, true));
  assert.deepEqual(stub.state.unloads, []);
  assert.deepEqual(stub.state.queueStops, []);
  assert.notEqual(outcome.status, "ejected");
});

test("declining to stop the chat leaves the model loaded", async () => {
  stub.resetStub({ active: QWEN, loaded: [QWEN], generating: true });
  stub.state.confirmAnswer = { proceed: false, forceCancelActive: false };
  const outcome = await ejectLoadedModel(chatRow(QWEN));
  assert.deepEqual(stub.state.unloads, []);
  assert.deepEqual(outcome, { status: "declined" });
  assert.equal(stub.state.active, QWEN);
});

test("an idle model is ejected without forcing anything", async () => {
  stub.resetStub({ active: QWEN, loaded: [QWEN] });
  const outcome = await ejectLoadedModel(chatRow(QWEN));
  assert.equal(stub.state.unloads[0].force_cancel_active ?? false, false);
  assert.deepEqual(outcome, { status: "ejected" });
});
