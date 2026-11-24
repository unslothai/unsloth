// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

const { store: localStorageFake } = installLocalStorageFake();
localStorageFake.set("unsloth_chat_settings_imported_to_studio_db", "true");
register("./store-settings-resolver.mjs", import.meta.url);

const { settingsHttp } = await import("./helpers/store-stubs/settings-http.ts");
const { useChatRuntimeStore } = await import(
  "../src/features/chat/stores/chat-runtime-store.ts"
);
const { servedCheckpoint, withoutNamedQuant } = await import(
  "../src/features/hub/lib/model-identity.ts"
);
const { chatModelLoadedTarget } = await import(
  "../src/features/chat/components/chat-model-notice-switch.ts"
);
const { resolveInferenceCheckpointId, servedInferenceCheckpointId } = await import(
  "../src/features/chat/lib/apply-inference-status-to-store.ts"
);

const REPO = "unsloth/SmolLM2-135M-Instruct-GGUF";
const Q4 = `${REPO}:Q4_K_M`;
const Q8 = `${REPO}:Q8_0`;
const OLLAMA = "ollama/llama3:8b";
const EXTERNAL = "external::anthropic::claude-opus-5";

const state = () => useChatRuntimeStore.getState();
const setTemperature = (temperature: number) =>
  state().setParams({ ...state().params, temperature });

async function settled(): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, 600));
}

test("a repo's settings survive its second quant loading, switching and unloading", async () => {
  settingsHttp.settings = {};
  await state().hydratePersistedSettings();
  state().setCheckpoint(REPO, "Q8_0");
  setTemperature(0.3);

  state().setCheckpoint(Q8, "Q8_0");
  assert.equal(state().params.checkpoint, REPO);
  assert.equal(state().params.temperature, 0.3);

  setTemperature(0.5);
  state().setCheckpoint(Q4, "Q4_K_M");
  assert.equal(state().params.checkpoint, REPO);
  assert.equal(state().params.temperature, 0.5);

  state().setCheckpoint(REPO, "Q4_K_M");
  assert.equal(state().params.temperature, 0.5);

  await settled();
  assert.deepEqual(Object.keys(state().paramsByModel), [REPO]);
  for (const put of settingsHttp.puts) {
    const byModel = (put.inferenceParamsByModel ?? {}) as Record<string, unknown>;
    assert.deepEqual(
      Object.keys(byModel).filter((key) => key !== REPO),
      [],
    );
  }
});

test("only a model's own quant is taken off its name", () => {
  assert.equal(withoutNamedQuant(Q8, "Q8_0"), REPO);
  assert.equal(withoutNamedQuant(Q8, "q8_0"), REPO);
  assert.equal(withoutNamedQuant(Q8, "Q4_K_M"), Q8);
  assert.equal(withoutNamedQuant(OLLAMA, null), OLLAMA);
  assert.equal(withoutNamedQuant(OLLAMA, "Q4_K_M"), OLLAMA);
  assert.equal(withoutNamedQuant(EXTERNAL, null), EXTERNAL);
  for (const name of [
    "ollama/llama3:Q8_0:Q8_0",
    "ollama-manifest:library/qwen3:Q8_0",
    "hf.co/org/A-GGUF:Q8_0",
    "llama3:Q8_0",
    "/models/org/A-GGUF:Q8_0",
    "C:\\models\\A:Q8_0",
    "ref:Q8_0",
  ]) {
    assert.equal(withoutNamedQuant(name, "Q8_0"), name);
    state().setCheckpoint(name, "Q8_0");
    assert.equal(state().params.checkpoint, name);
  }
  state().setCheckpoint(OLLAMA, null);
  assert.equal(state().params.checkpoint, OLLAMA);
  state().setCheckpoint(EXTERNAL);
  assert.equal(state().params.checkpoint, EXTERNAL);
});

test("status selects the repo and serves the copy", () => {
  const beside = { active_model: REPO, model_identifier: Q8, gguf_variant: "Q8_0" } as never;
  assert.equal(resolveInferenceCheckpointId(beside), REPO);
  assert.equal(servedInferenceCheckpointId(beside), Q8);
  const alone = { active_model: REPO, model_identifier: REPO, gguf_variant: "Q8_0" } as never;
  assert.equal(resolveInferenceCheckpointId(alone), REPO);
  const ollama = { active_model: OLLAMA, model_identifier: OLLAMA, gguf_variant: null } as never;
  assert.equal(resolveInferenceCheckpointId(ollama), OLLAMA);
  assert.equal(resolveInferenceCheckpointId({ active_model: null } as never), null);
});

test("requests name the selected copy only while two quants of the repo are loaded", () => {
  const both = [
    { id: REPO, quant: "Q8_0", checkpoint: Q8 },
    { id: REPO, quant: "Q4_K_M", checkpoint: Q4 },
    { id: "unsloth/Other-GGUF", quant: "Q8_0", checkpoint: "unsloth/Other-GGUF" },
  ];
  assert.equal(servedCheckpoint(both, REPO, "Q4_K_M"), Q4);
  assert.equal(servedCheckpoint(both, REPO, "q8_0"), Q8);
  assert.equal(servedCheckpoint(both, REPO, "Q2_K"), REPO);
  assert.equal(servedCheckpoint(both, REPO, null), REPO);
  assert.equal(servedCheckpoint(both, "unsloth/Other-GGUF", "Q8_0"), "unsloth/Other-GGUF");
  const alone = [{ id: REPO, quant: "Q8_0", checkpoint: REPO }];
  assert.equal(servedCheckpoint(alone, REPO, "Q8_0"), REPO);
});

test("a chat goes back to its own quant when two of its repo are loaded", () => {
  const slots = [
    { id: REPO, quant: "Q8_0", checkpoint: Q8 },
    { id: REPO, quant: "Q4_K_M", checkpoint: Q4 },
  ];
  assert.deepEqual(
    chatModelLoadedTarget({ modelId: REPO, ggufVariant: "Q4_K_M" }, slots),
    { modelId: REPO, ggufVariant: "Q4_K_M" },
  );
});
