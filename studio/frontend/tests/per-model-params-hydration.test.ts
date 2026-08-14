// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Startup order decides whether the per-model memory survives. The inference
// status can land before the settings response, and the model it reports was
// never switched to, so nothing replays its memory on its own. These drive the
// real store through that order and through a steady-state poll.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

const { store: localStorageFake } = installLocalStorageFake();
// Skip the legacy import path: it would look for settings this test never wrote.
localStorageFake.set("unsloth_chat_settings_imported_to_studio_db", "true");
register("./store-settings-resolver.mjs", import.meta.url);

const { settingsHttp } = await import("./helpers/store-stubs/settings-http.ts");
const { useChatRuntimeStore } = await import(
  "../src/features/chat/stores/chat-runtime-store.ts"
);
const { mergeBackendRecommendedInference } = await import(
  "../src/features/chat/presets/preset-policy.ts"
);

const QWEN = "unsloth/Qwen3.5-9B-GGUF";
const LLAMA = "unsloth/Llama-4-8B";
const TUNED = { temperature: 0.2, maxTokens: 4096, systemPrompt: "Be terse." };

/** A status response for a resident GGUF, recommending its own sampling. */
const STATUS = {
  inference: { temperature: 0.9, top_p: 0.5 },
  is_gguf: true,
  context_length: 131072,
} as never;

/** applyActiveModelStatusToStore's update, which the last test pins. */
function applyStatus(modelId: string) {
  const store = useChatRuntimeStore.getState();
  store.setParams(
    mergeBackendRecommendedInference({
      current: store.params,
      response: STATUS,
      modelId,
      presetSource: store.activePresetSource,
    }),
    { fromModelDefaults: true },
  );
}

/** The debounced settings writer, flushed. */
async function settled(): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, 600));
}

test("a status response that beats hydration keeps the model's settings", async () => {
  settingsHttp.settings = {
    inferenceParams: TUNED,
    inferenceParamsByModel: { [QWEN]: TUNED },
  };
  settingsHttp.hold();
  const hydrating = useChatRuntimeStore.getState().hydratePersistedSettings();

  applyStatus(QWEN);
  // Nothing is recorded before hydration: these params are the recommendation
  // the backend just sent, not settings this model was used with.
  assert.deepEqual(useChatRuntimeStore.getState().paramsByModel, {});

  settingsHttp.release?.();
  await hydrating;

  const hydrated = useChatRuntimeStore.getState();
  assert.deepEqual(
    hydrated.paramsByModel[QWEN],
    TUNED,
    "the persisted entry is not fenced out by the status update",
  );
  // The status set the global params, and a model that was already resident
  // never crosses a checkpoint transition, so hydration is the only replay.
  assert.equal(hydrated.params.temperature, 0.2);
  assert.equal(hydrated.params.maxTokens, 4096);
  assert.equal(hydrated.params.systemPrompt, "Be terse.");
  // Params this model never pinned still take the recommendation.
  assert.equal(hydrated.params.topP, 0.5);

  // The reported failure was durable: switching away wrote the recommendation
  // over the tuning, so it was gone on the next launch too.
  settingsHttp.puts.length = 0;
  useChatRuntimeStore
    .getState()
    .setParams({ ...useChatRuntimeStore.getState().params, checkpoint: LLAMA });
  await settled();
  const written = settingsHttp.puts.at(-1)?.inferenceParamsByModel as Record<
    string,
    Record<string, unknown>
  >;
  assert.equal(written[QWEN].temperature, 0.2);
  assert.equal(written[QWEN].maxTokens, 4096);
});

// A status poll re-applies the recommendation on every refresh, so without
// laying the memory back over it the tuning lasts only until the next poll.
test("a status poll does not undo the model's remembered settings", () => {
  useChatRuntimeStore.setState({
    params: {
      ...useChatRuntimeStore.getState().params,
      checkpoint: QWEN,
      temperature: 0.2,
    },
    paramsByModel: { [QWEN]: TUNED },
  });

  applyStatus(QWEN);

  const after = useChatRuntimeStore.getState();
  assert.equal(after.params.temperature, 0.2);
  assert.equal(after.params.maxTokens, 4096);
});

// A model with nothing remembered must still take the recommendation, or the
// memory would just be the old global set under a new name.
test("a model with nothing remembered still takes the recommendation", () => {
  useChatRuntimeStore.setState({
    params: { ...useChatRuntimeStore.getState().params, checkpoint: LLAMA },
    paramsByModel: {},
  });

  applyStatus(LLAMA);

  assert.equal(useChatRuntimeStore.getState().params.temperature, 0.9);
});

// The three places that re-apply a model's defaults. Each overwrites remembered
// values without changing the checkpoint, so each has to ask for the replay;
// they pull in the chat UI, so this reads them rather than importing them.
test("every site that re-applies model defaults asks for the replay", () => {
  const sites: [string, RegExp][] = [
    [
      "../src/features/chat/lib/apply-inference-status-to-store.ts",
      /mergeBackendRecommendedInference\([\s\S]{0,400}?\{ fromModelDefaults: true \}/,
    ],
    [
      "../src/features/chat/hooks/use-chat-model-runtime.ts",
      /mergeBackendRecommendedInference\([\s\S]{0,400}?\{ fromModelDefaults: true \}/,
    ],
    [
      // The Qwen3 thinking-mode params applied after a load.
      "../src/features/chat/hooks/use-chat-model-runtime.ts",
      /setParams\(\s*\{ \.\.\.store\.params, \.\.\.p \},\s*\{ fromModelDefaults: true \},/,
    ],
  ];
  for (const [path, pattern] of sites) {
    const source = readFileSync(new URL(path, import.meta.url), "utf8");
    assert.match(source, pattern, path);
  }
});
