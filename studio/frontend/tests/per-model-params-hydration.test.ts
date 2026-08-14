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

// A pre-hydration edit is the user's, and the fence that protects it from the
// hydrated global set has to protect it from the replay too.
test("a pre-hydration edit outranks the replay", async () => {
  settingsHttp.settings = {
    inferenceParams: { temperature: 0.2, systemPrompt: "Be terse." },
    inferenceParamsByModel: { [QWEN]: TUNED },
  };
  settingsHttp.hold();
  useChatRuntimeStore.setState({
    params: { ...useChatRuntimeStore.getState().params, checkpoint: QWEN },
    paramsByModel: {},
    // Hydration runs once per store, so re-arm it for a second startup.
    settingsHydrated: false,
  });
  const hydrating = useChatRuntimeStore.getState().hydratePersistedSettings();

  const store = useChatRuntimeStore.getState();
  store.setParams({ ...store.params, temperature: 0.85 });

  settingsHttp.release?.();
  await hydrating;

  const params = useChatRuntimeStore.getState().params;
  assert.equal(params.temperature, 0.85, "the slider the user just moved");
  assert.equal(
    params.systemPrompt,
    "Be terse.",
    "a key the user did not touch still replays",
  );
});

// A stored entry can be partial: an older write, or a field that did not survive
// sanitising. Replay lays what it has over the params on screen, so the gaps
// would come from whichever model happened to be selected beforehand.
test("a partial stored entry does not borrow from the previous model", async () => {
  settingsHttp.settings = {
    inferenceParams: { temperature: 0.5, topP: 0.9, systemPrompt: "saved" },
    // Only one field, as an older client or a hand-written payload would leave it.
    inferenceParamsByModel: { [QWEN]: { temperature: 0.15 } },
  };
  useChatRuntimeStore.setState({
    params: {
      ...useChatRuntimeStore.getState().params,
      checkpoint: LLAMA,
      topP: 0.11,
      systemPrompt: "the other model's",
    },
    paramsByModel: {},
    settingsHydrated: false,
  });
  await useChatRuntimeStore.getState().hydratePersistedSettings();

  const store = useChatRuntimeStore.getState();
  assert.equal(
    store.paramsByModel[QWEN]?.topP,
    0.9,
    "filled from the saved set",
  );
  store.setParams({ ...store.params, checkpoint: QWEN });

  const params = useChatRuntimeStore.getState().params;
  assert.equal(params.temperature, 0.15, "what the entry does hold");
  assert.equal(params.topP, 0.9);
  assert.equal(
    params.systemPrompt,
    "saved",
    "not the prompt the previous model was using",
  );
});

// Staging writes the next model's context length into the params of the one
// still on screen, and the switch that follows snapshots those params. Marking
// the staging call alone left the value to reach the outgoing model anyway.
test("the model being left is remembered without the staged value", () => {
  useChatRuntimeStore.setState({
    params: {
      ...useChatRuntimeStore.getState().params,
      checkpoint: LLAMA,
      maxSeqLength: 4096,
      temperature: 0.33,
    },
    paramsByModel: {},
  });

  // applyPerModelConfigToRuntime, staging the model about to load.
  const staging = useChatRuntimeStore.getState();
  staging.setParams(
    { ...staging.params, maxSeqLength: 32768 },
    { stagedForLoad: true },
  );
  // The load lands and the checkpoint moves.
  const switching = useChatRuntimeStore.getState();
  switching.setParams(
    { ...switching.params, checkpoint: QWEN },
    { fromModelDefaults: true },
  );

  const remembered = useChatRuntimeStore.getState().paramsByModel[LLAMA];
  assert.equal(remembered?.maxSeqLength, 4096, "its own context, not Qwen's");
  assert.equal(remembered?.temperature, 0.33, "everything else still recorded");
});

// Only what staging wrote, and only while nothing has changed it since.
test("an edit after staging outranks the staged value", () => {
  useChatRuntimeStore.setState({
    params: {
      ...useChatRuntimeStore.getState().params,
      checkpoint: LLAMA,
      maxSeqLength: 4096,
    },
    paramsByModel: {},
  });

  const staging = useChatRuntimeStore.getState();
  staging.setParams(
    { ...staging.params, maxSeqLength: 32768 },
    { stagedForLoad: true },
  );
  // The user sets a context of their own before the load happens.
  const editing = useChatRuntimeStore.getState();
  editing.setParams({ ...editing.params, maxSeqLength: 16384 });
  const switching = useChatRuntimeStore.getState();
  switching.setParams({ ...switching.params, checkpoint: QWEN });

  assert.equal(
    useChatRuntimeStore.getState().paramsByModel[LLAMA]?.maxSeqLength,
    16384,
  );
});

// A model's defaults are not settings it was used with. Recording them makes
// the next defaults hook replay them over itself: a fresh Qwen3 load applied
// the load response, recorded it, and then the thinking-mode params were
// immediately replaced by what had just been recorded.
test("model defaults are replayed over, not recorded", () => {
  useChatRuntimeStore.setState({
    params: { ...useChatRuntimeStore.getState().params, checkpoint: LLAMA },
    paramsByModel: {},
  });

  applyStatus(QWEN);
  assert.equal(
    useChatRuntimeStore.getState().paramsByModel[QWEN],
    undefined,
    "the recommendation is not memory",
  );

  // The Qwen3 thinking params, applied straight after the load response.
  const store = useChatRuntimeStore.getState();
  store.setParams(
    { ...store.params, temperature: 0.6, minP: 0, presencePenalty: 1.5 },
    { fromModelDefaults: true },
  );
  const params = useChatRuntimeStore.getState().params;
  assert.equal(params.temperature, 0.6);
  assert.equal(params.minP, 0);
  assert.equal(params.presencePenalty, 1.5);
});

// Params staged for the model about to load are applied while the previous one
// is still current, so they must not be filed against it.
test("staged load params are not filed against the model on screen", () => {
  useChatRuntimeStore.setState({
    params: {
      ...useChatRuntimeStore.getState().params,
      checkpoint: LLAMA,
      maxSeqLength: 4096,
    },
    paramsByModel: {},
  });

  const store = useChatRuntimeStore.getState();
  store.setParams(
    { ...store.params, maxSeqLength: 32768 },
    { stagedForLoad: true },
  );

  const after = useChatRuntimeStore.getState();
  assert.equal(
    after.params.maxSeqLength,
    32768,
    "still applied to the runtime",
  );
  assert.deepEqual(after.paramsByModel, {}, "but not remembered for Llama");
});

// Unloading or evicting leaves a model the same way switching does.
test("clearing the checkpoint remembers the model being dropped", () => {
  useChatRuntimeStore.setState({
    params: {
      ...useChatRuntimeStore.getState().params,
      checkpoint: LLAMA,
      temperature: 0.11,
    },
    paramsByModel: {},
  });

  useChatRuntimeStore.getState().clearCheckpoint();

  assert.equal(
    useChatRuntimeStore.getState().paramsByModel[LLAMA]?.temperature,
    0.11,
  );
});

// Lowering a GGUF's context and reloading: the remembered budget no longer fits.
test("a remembered budget is clamped to the context just loaded", () => {
  useChatRuntimeStore.setState({
    params: {
      ...useChatRuntimeStore.getState().params,
      checkpoint: QWEN,
      maxTokens: 8192,
    },
    paramsByModel: { [QWEN]: { maxTokens: 131072 } },
  });

  const store = useChatRuntimeStore.getState();
  store.setParams(
    { ...store.params, maxTokens: 8192 },
    { fromModelDefaults: true, maxTokensCap: 8192 },
  );

  assert.equal(useChatRuntimeStore.getState().params.maxTokens, 8192);
});

// The three places that re-apply a model's defaults. Each overwrites remembered
// values without changing the checkpoint, so each has to ask for the replay;
// they pull in the chat UI, so this reads them rather than importing them.
test("every site that re-applies model defaults asks for the replay", () => {
  const sites: [string, RegExp][] = [
    [
      "../src/features/chat/lib/apply-inference-status-to-store.ts",
      /mergeBackendRecommendedInference\([\s\S]{0,500}?fromModelDefaults: true/,
    ],
    [
      "../src/features/chat/hooks/use-chat-model-runtime.ts",
      /mergeBackendRecommendedInference\([\s\S]{0,500}?fromModelDefaults: true/,
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

// A GGUF reloaded with a smaller context must not keep a budget remembered from
// a larger one: the request would exceed what was actually loaded.
test("the sites that know the loaded context pass it as the cap", () => {
  const caps: [string, RegExp][] = [
    [
      "../src/features/chat/lib/apply-inference-status-to-store.ts",
      /maxTokensCap: status\.is_gguf/,
    ],
    [
      "../src/features/chat/hooks/use-chat-model-runtime.ts",
      /maxTokensCap: loadResponse\.is_gguf/,
    ],
    ["../src/features/chat/api/chat-adapter.ts", /maxTokensCap:/],
  ];
  for (const [path, pattern] of caps) {
    const source = readFileSync(new URL(path, import.meta.url), "utf8");
    assert.match(source, pattern, path);
  }
});

// Staged load params describe the model about to load, so they must not be
// filed against the one still on screen.
test("a staged per-model config is marked as such", () => {
  const source = readFileSync(
    new URL(
      "../src/features/model-picker/model-config/apply-per-model-config.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(
    source,
    /setParams\(\s*\{ \.\.\.store\.params, maxSeqLength \},\s*\{ stagedForLoad: true \}/,
  );
});
