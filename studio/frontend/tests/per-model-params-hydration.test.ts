// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Inference status can land before settings, and that model never switched, so nothing replays.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake, readSrc } from "./helpers/kit.ts";

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
const { DEFAULT_INFERENCE_PARAMS } = await import(
  "../src/features/chat/types/runtime.ts"
);

const QWEN = "unsloth/Qwen3.5-9B-GGUF";
const LLAMA = "unsloth/Llama-4-8B";
const EXTERNAL = "external::anthropic::claude-opus-5";
const TUNED = { temperature: 0.2, maxTokens: 4096, systemPrompt: "Be terse." };

const STATUS_CONTEXT_LENGTH = 131072;
const STATUS = {
  inference: { temperature: 0.9, top_p: 0.5 },
  is_gguf: true,
  context_length: STATUS_CONTEXT_LENGTH,
} as never;

/** applyActiveModelStatusToStore's update; its source is pinned by the re-apply sites test. */
function applyStatus(
  modelId: string,
  { adoptingExistingServerModel = false } = {},
) {
  const store = useChatRuntimeStore.getState();
  store.setParams(
    mergeBackendRecommendedInference({
      current: store.params,
      response: STATUS,
      modelId,
      presetSource: store.activePresetSource,
      loadedContextLength: STATUS_CONTEXT_LENGTH,
    }),
    {
      fromModelDefaults: true,
      migrateOwnedGlobalQwenDefaults: adoptingExistingServerModel,
    },
  );
}

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
  assert.deepEqual(useChatRuntimeStore.getState().paramsByModel, {});

  settingsHttp.release?.();
  await hydrating;

  const hydrated = useChatRuntimeStore.getState();
  assert.deepEqual(
    hydrated.paramsByModel[QWEN],
    TUNED,
    "the persisted entry is not fenced out by the status update",
  );
  assert.equal(hydrated.params.temperature, 0.2);
  assert.equal(hydrated.params.maxTokens, 4096);
  assert.equal(hydrated.params.systemPrompt, "Be terse.");
  assert.equal(hydrated.params.topP, 0.5);

  settingsHttp.puts.length = 0;
  useChatRuntimeStore
    .getState()
    .setParams({ ...useChatRuntimeStore.getState().params, checkpoint: LLAMA });
  await settled();
  for (const put of settingsHttp.puts) {
    assert.equal(
      (put.inferenceParamsByModel as Record<string, unknown>)?.[QWEN],
      undefined,
      "the recommendation is not written over the tuning",
    );
  }
  const held = useChatRuntimeStore.getState().paramsByModel[QWEN];
  assert.equal(held?.temperature, 0.2, "the tuning this browser still holds");
  assert.equal(held?.maxTokens, 4096);
  assert.equal(held?.systemPrompt, "Be terse.");
});

// A status poll re-applies the recommendation on every refresh.
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

test("a model with nothing remembered still takes the recommendation", () => {
  useChatRuntimeStore.setState({
    params: { ...useChatRuntimeStore.getState().params, checkpoint: LLAMA },
    paramsByModel: {},
  });

  applyStatus(LLAMA);

  assert.equal(useChatRuntimeStore.getState().params.temperature, 0.9);
});

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

test("a partial stored entry is neither filled nor borrowed from", async () => {
  settingsHttp.settings = {
    inferenceParams: { temperature: 0.5, topP: 0.9, systemPrompt: "saved" },
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

  assert.deepEqual(
    useChatRuntimeStore.getState().paramsByModel[QWEN],
    { temperature: 0.15 },
    "stored as written, not grown with another model's settings",
  );

  const store = useChatRuntimeStore.getState();
  store.setParams(
    { ...store.params, checkpoint: QWEN, topP: 0.8, systemPrompt: "" },
    { fromModelDefaults: true },
  );

  const params = useChatRuntimeStore.getState().params;
  assert.equal(params.temperature, 0.15, "what the entry does hold");
  assert.equal(params.topP, 0.8, "the gap takes this model's own default");
  assert.equal(
    params.systemPrompt,
    "",
    "not the prompt the previous model was using",
  );
});

test("the context length is not part of what a model remembers", () => {
  useChatRuntimeStore.setState({
    params: {
      ...useChatRuntimeStore.getState().params,
      checkpoint: LLAMA,
      maxSeqLength: 4096,
      temperature: 0.33,
    },
    paramsByModel: {},
  });

  const staging = useChatRuntimeStore.getState();
  staging.setParams({ ...staging.params, maxSeqLength: 32768 });
  assert.deepEqual(
    useChatRuntimeStore.getState().paramsByModel,
    {},
    "a context on its own is not an edit this remembers",
  );

  const switching = useChatRuntimeStore.getState();
  switching.setParams(
    { ...switching.params, checkpoint: QWEN },
    { fromModelDefaults: true },
  );

  const remembered = useChatRuntimeStore.getState().paramsByModel[LLAMA];
  assert.equal(remembered?.temperature, 0.33, "its sampling is remembered");
  assert.equal(
    "maxSeqLength" in (remembered ?? {}),
    false,
    "its context is not, so nothing replays over the loaded one",
  );
});

test("a model loaded before hydration keeps its own defaults", async () => {
  settingsHttp.settings = {
    inferenceParams: { temperature: 0.42, systemPrompt: "the last model's" },
    inferenceParamsByModel: {},
  };
  settingsHttp.hold();
  useChatRuntimeStore.setState({
    params: { ...useChatRuntimeStore.getState().params, checkpoint: LLAMA },
    paramsByModel: {},
    settingsHydrated: false,
  });
  const hydrating = useChatRuntimeStore.getState().hydratePersistedSettings();

  applyStatus(QWEN);

  settingsHttp.release?.();
  await hydrating;

  const params = useChatRuntimeStore.getState().params;
  assert.equal(
    params.temperature,
    0.9,
    "the recommendation it loaded with, not the saved global set",
  );
  assert.equal(params.topP, 0.5);
});

test("the resident model keeps the settings saved for it", async () => {
  settingsHttp.settings = {
    inferenceParams: { temperature: 0.2, systemPrompt: "tuned" },
  };
  settingsHttp.hold();
  useChatRuntimeStore.setState({
    params: {
      ...useChatRuntimeStore.getState().params,
      checkpoint: "",
      temperature: 0.5,
    },
    paramsByModel: {},
    settingsHydrated: false,
  });
  const hydrating = useChatRuntimeStore.getState().hydratePersistedSettings();

  applyStatus(QWEN, { adoptingExistingServerModel: true });

  settingsHttp.release?.();
  await hydrating;

  const params = useChatRuntimeStore.getState().params;
  assert.equal(
    params.temperature,
    0.2,
    "the saved value, not the recommendation",
  );
  assert.equal(params.systemPrompt, "tuned");
});

test("a restore does not remember the model a hidden load left", () => {
  useChatRuntimeStore.setState({
    params: {
      ...useChatRuntimeStore.getState().params,
      checkpoint: QWEN,
      temperature: 0.77,
    },
    paramsByModel: {},
  });

  useChatRuntimeStore.getState().setCheckpoint(LLAMA, undefined, {
    trackQueuedSettings: false,
    persist: false,
  });

  assert.deepEqual(useChatRuntimeStore.getState().paramsByModel, {});
});

test("a visible switch remembers the model being left", () => {
  useChatRuntimeStore.setState({
    params: {
      ...useChatRuntimeStore.getState().params,
      checkpoint: QWEN,
      temperature: 0.77,
    },
    paramsByModel: {},
  });

  useChatRuntimeStore.getState().setCheckpoint(LLAMA);

  assert.equal(
    useChatRuntimeStore.getState().paramsByModel[QWEN]?.temperature,
    0.77,
  );
});

test("a default equal to the previous model's value is still kept", async () => {
  settingsHttp.settings = {
    inferenceParams: { temperature: 0.2 },
    inferenceParamsByModel: {},
  };
  settingsHttp.hold();
  useChatRuntimeStore.setState({
    params: {
      ...useChatRuntimeStore.getState().params,
      checkpoint: LLAMA,
      temperature: 0.9,
    },
    paramsByModel: {},
    settingsHydrated: false,
  });
  const hydrating = useChatRuntimeStore.getState().hydratePersistedSettings();

  applyStatus(QWEN);

  settingsHttp.release?.();
  await hydrating;

  assert.equal(
    useChatRuntimeStore.getState().params.temperature,
    0.9,
    "the model's own default, not the other model's saved value",
  );
});

test("the replay at hydration fits the context already published", async () => {
  settingsHttp.settings = {
    inferenceParams: {},
    inferenceParamsByModel: { [QWEN]: { maxTokens: 131072 } },
  };
  settingsHttp.hold();
  useChatRuntimeStore.setState({
    params: {
      ...useChatRuntimeStore.getState().params,
      checkpoint: QWEN,
      maxTokens: 8192,
    },
    paramsByModel: {},
    loadedContextLength: 8192,
    settingsHydrated: false,
  });
  const hydrating = useChatRuntimeStore.getState().hydratePersistedSettings();
  settingsHttp.release?.();
  await hydrating;

  assert.equal(useChatRuntimeStore.getState().params.maxTokens, 8192);
});

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

// These sites pull in the chat UI, so read their source instead of importing.
test("every site that re-applies model defaults asks for the replay", () => {
  // Window sized to stay short of the next fromModelDefaults site in either file.
  const sites: [string, RegExp][] = [
    [
      "../src/features/chat/lib/apply-inference-status-to-store.ts",
      /mergeBackendRecommendedInference\([\s\S]{0,1200}?fromModelDefaults: true/,
    ],
    [
      "../src/features/chat/hooks/use-chat-model-runtime.ts",
      /mergeBackendRecommendedInference\([\s\S]{0,1200}?fromModelDefaults: true/,
    ],
    [
      "../src/features/chat/hooks/use-chat-model-runtime.ts",
      /setParams\(\{ \.\.\.store\.params, \.\.\.p \}, \{\s*fromModelDefaults: true,/,
    ],
  ];
  for (const [path, pattern] of sites) {
    const source = readFileSync(new URL(path, import.meta.url), "utf8");
    assert.match(source, pattern, path);
  }
});

test("an edit made before hydration is kept by the model's entry", async () => {
  useChatRuntimeStore.setState({
    settingsHydrated: false,
    rememberParamsPerModel: true,
    paramsByModel: {},
    params: { ...useChatRuntimeStore.getState().params, checkpoint: QWEN },
  });
  settingsHttp.settings = {
    inferenceParams: { temperature: 0.9 },
    inferenceParamsByModel: {
      [QWEN]: { temperature: 0.9, systemPrompt: "stale" },
    },
  };
  settingsHttp.hold();
  const hydrating = useChatRuntimeStore.getState().hydratePersistedSettings();

  const editing = useChatRuntimeStore.getState();
  editing.setParams({ ...editing.params, temperature: 0.33 });

  settingsHttp.release?.();
  await hydrating;

  const hydrated = useChatRuntimeStore.getState();
  assert.equal(hydrated.params.temperature, 0.33, "the fence held");
  assert.equal(
    hydrated.paramsByModel[QWEN]?.temperature,
    0.33,
    "and the entry took the edit rather than the value it was written before",
  );
  assert.equal(hydrated.paramsByModel[QWEN]?.systemPrompt, "stale");

  applyStatus(QWEN);
  assert.equal(
    useChatRuntimeStore.getState().params.temperature,
    0.33,
    "so a poll that re-applies defaults replays the edit, not the old value",
  );
});

test("a remembered budget is capped by a non-GGUF load", () => {
  const runtime = readSrc("features/chat/hooks/use-chat-model-runtime.ts");
  // The reported window leads; a self-sizing backend gets the auto-size sentinel.
  assert.match(
    runtime,
    /const loadedContextCap = replayMaxTokensCap\(\s*loadedFields\.loadedContextLength \?\?\s*\(!loadResponse\.is_gguf && effectiveMaxSeqLength > 0\s*\? effectiveMaxSeqLength\s*: null\),\s*\);/,
  );
  assert.equal(
    runtime.match(/maxTokensCap: loadedContextCap/g)?.length,
    2,
    "the thinking-defaults replay is capped too",
  );

  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  assert.match(
    adapter,
    /maxTokensCap: replayMaxTokensCap\(\s*candidate\.kind === "gguf"\s*\? loadedContextFields\(loadResp\)\.loadedContextLength\s*: loadedWindow,\s*\),/,
  );

  // A pane with no context pin sends the sentinel; capping at 0 would ask for no output.
  const composer = readSrc("features/chat/shared-composer.tsx");
  assert.match(
    composer,
    /maxTokensCap: replayMaxTokensCap\(\s*loadedContextFields\(resp\)\.loadedContextLength \?\?\s*\(!resp\.is_gguf && effectiveMaxSeqLength > 0/,
  );

  const status = readSrc("features/chat/lib/apply-inference-status-to-store.ts");
  assert.match(status, /maxTokensCap: replayMaxTokensCap\(status\.context_length\),/);
});

test("the cap wins over the remembered budget", () => {
  useChatRuntimeStore.setState({
    settingsHydrated: true,
    rememberParamsPerModel: true,
    paramsByModel: { [LLAMA]: { maxTokens: 32768 } },
    params: { ...useChatRuntimeStore.getState().params, checkpoint: LLAMA },
  });
  const store = useChatRuntimeStore.getState();
  store.setParams(
    { ...store.params, maxSeqLength: 8192, maxTokens: 8192 },
    { fromModelDefaults: true, maxTokensCap: 8192 },
  );
  assert.equal(useChatRuntimeStore.getState().params.maxTokens, 8192);

  const uncapped = useChatRuntimeStore.getState();
  uncapped.setParams(
    { ...uncapped.params, maxTokens: 8192 },
    { fromModelDefaults: true },
  );
  assert.equal(useChatRuntimeStore.getState().params.maxTokens, 32768);
});

// A mirrored scalar setting, so it persists via setScalarSettingVersion.
test("turning the memory off is persisted and hydrated back", async () => {
  useChatRuntimeStore.setState({
    settingsHydrated: true,
    rememberParamsPerModel: true,
  });
  settingsHttp.puts.length = 0;
  useChatRuntimeStore.getState().setRememberParamsPerModel(false);
  await settled();
  assert.equal(
    settingsHttp.puts.at(-1)?.rememberParamsPerModel,
    false,
    "the choice is written, not just held in the store",
  );

  useChatRuntimeStore.setState({
    settingsHydrated: false,
    rememberParamsPerModel: true,
  });
  settingsHttp.settings = { rememberParamsPerModel: false };
  await useChatRuntimeStore.getState().hydratePersistedSettings();
  assert.equal(useChatRuntimeStore.getState().rememberParamsPerModel, false);
});

// A backend that sizes no window leaves loadedContextLength null.
test("a safetensors context also caps the hydration replay", async () => {
  useChatRuntimeStore.setState({
    settingsHydrated: false,
    rememberParamsPerModel: true,
    loadedContextLength: null,
    paramsByModel: {},
    params: { ...useChatRuntimeStore.getState().params, checkpoint: LLAMA },
  });
  settingsHttp.settings = {
    inferenceParams: { maxTokens: 32768 },
    inferenceParamsByModel: { [LLAMA]: { maxTokens: 32768 } },
  };
  settingsHttp.hold();
  const hydrating = useChatRuntimeStore.getState().hydratePersistedSettings();

  const store = useChatRuntimeStore.getState();
  store.setParams(
    { ...store.params, maxSeqLength: 8192, maxTokens: 8192 },
    { fromModelDefaults: true, maxTokensCap: 8192 },
  );
  assert.equal(useChatRuntimeStore.getState().params.maxTokens, 8192);

  settingsHttp.release?.();
  await hydrating;
  assert.equal(
    useChatRuntimeStore.getState().params.maxTokens,
    8192,
    "the replay fits the context the load actually has",
  );
});

test("a kept context does not follow the next model", async () => {
  useChatRuntimeStore.setState({
    settingsHydrated: false,
    rememberParamsPerModel: true,
    loadedContextLength: null,
    paramsByModel: {},
    params: { ...useChatRuntimeStore.getState().params, checkpoint: LLAMA },
  });
  settingsHttp.settings = {
    inferenceParams: { maxTokens: 32768 },
    inferenceParamsByModel: { [QWEN]: { maxTokens: 32768 } },
  };
  settingsHttp.hold();
  const hydrating = useChatRuntimeStore.getState().hydratePersistedSettings();

  const store = useChatRuntimeStore.getState();
  store.setParams(
    { ...store.params, maxTokens: 8192 },
    { fromModelDefaults: true, maxTokensCap: 8192 },
  );
  const switched = useChatRuntimeStore.getState();
  switched.setParams({ ...switched.params, checkpoint: QWEN });

  settingsHttp.release?.();
  await hydrating;
  assert.equal(
    useChatRuntimeStore.getState().params.maxTokens,
    32768,
    "the other model's smaller context does not clamp this one",
  );
});

test("turning the memory off keeps the settings on screen", async () => {
  useChatRuntimeStore.setState({
    settingsHydrated: true,
    rememberParamsPerModel: true,
    paramsByModel: { [LLAMA]: { temperature: 0.11, systemPrompt: "B" } },
    params: {
      ...useChatRuntimeStore.getState().params,
      checkpoint: QWEN,
      temperature: 0.9,
      systemPrompt: "A",
    },
  });
  await settled();
  settingsHttp.puts.length = 0;

  const store = useChatRuntimeStore.getState();
  store.setParams(
    { ...store.params, checkpoint: LLAMA },
    { fromModelDefaults: true, persist: false },
  );
  assert.equal(useChatRuntimeStore.getState().params.temperature, 0.11);
  assert.equal(
    settingsHttp.puts.length,
    0,
    "the hidden restore wrote nothing, which is the point",
  );

  useChatRuntimeStore.getState().setRememberParamsPerModel(false);
  await settled();
  const written: Record<string, unknown> = {};
  for (const put of settingsHttp.puts) Object.assign(written, put);
  const globals = written.inferenceParams as Record<string, unknown>;
  assert.equal(globals?.temperature, 0.11);
  assert.equal(globals?.systemPrompt, "B");
});

test("the loaded context caps a global budget with no entry to replay", async () => {
  useChatRuntimeStore.setState({
    settingsHydrated: false,
    rememberParamsPerModel: true,
    loadedContextLength: null,
    paramsByModel: {},
    params: { ...useChatRuntimeStore.getState().params, checkpoint: LLAMA },
  });
  settingsHttp.settings = { inferenceParams: { maxTokens: 32768 } };
  settingsHttp.hold();
  const hydrating = useChatRuntimeStore.getState().hydratePersistedSettings();

  const store = useChatRuntimeStore.getState();
  store.setParams(
    { ...store.params, maxSeqLength: 8192, maxTokens: 8192 },
    { fromModelDefaults: true, maxTokensCap: 8192 },
  );

  settingsHttp.release?.();
  await hydrating;
  assert.equal(useChatRuntimeStore.getState().params.maxTokens, 8192);
});

// The server merges per key, so a full snapshot would rewrite every field.
test("a browser that only read an entry does not write it back", async () => {
  useChatRuntimeStore.setState({
    settingsHydrated: false,
    rememberParamsPerModel: true,
    paramsByModel: {},
  });
  settingsHttp.settings = {
    inferenceParamsByModel: {
      [QWEN]: { temperature: 0.6 },
      [LLAMA]: { temperature: 0.7 },
    },
  };
  await useChatRuntimeStore.getState().hydratePersistedSettings();
  useChatRuntimeStore.setState({
    params: {
      ...useChatRuntimeStore.getState().params,
      checkpoint: QWEN,
      temperature: 0.6,
    },
  });
  await settled();

  const perModelWrites = async (): Promise<string[]> => {
    await settled();
    const keys = new Set<string>();
    for (const put of settingsHttp.puts) {
      for (const id of Object.keys(
        (put.inferenceParamsByModel ?? {}) as object,
      )) {
        keys.add(id);
      }
    }
    settingsHttp.puts.length = 0;
    return [...keys];
  };
  await perModelWrites();

  for (const checkpoint of [LLAMA, QWEN, LLAMA]) {
    const store = useChatRuntimeStore.getState();
    store.setParams({ ...store.params, checkpoint });
    assert.deepEqual(
      await perModelWrites(),
      [],
      "a switch reads the entries, it does not rewrite them",
    );
  }
  assert.equal(useChatRuntimeStore.getState().params.temperature, 0.7);

  // Only the moved key is sent, so another tab's changes are not overwritten.
  settingsHttp.puts.length = 0;
  const editing = useChatRuntimeStore.getState();
  editing.setParams({ ...editing.params, temperature: 0.42 });
  await settled();
  const patch: Record<string, Record<string, unknown>> = {};
  for (const put of settingsHttp.puts) {
    Object.assign(
      patch,
      (put.inferenceParamsByModel ?? {}) as Record<
        string,
        Record<string, unknown>
      >,
    );
  }
  settingsHttp.puts.length = 0;
  assert.deepEqual(patch, { [LLAMA]: { temperature: 0.42 } });

  const leaving = useChatRuntimeStore.getState();
  leaving.setParams({ ...leaving.params, checkpoint: QWEN });
  assert.deepEqual(await perModelWrites(), []);
});

test("a model with no entry is still seeded when it is left", async () => {
  useChatRuntimeStore.setState({
    settingsHydrated: true,
    rememberParamsPerModel: true,
    paramsByModel: {},
    params: {
      ...useChatRuntimeStore.getState().params,
      checkpoint: QWEN,
      temperature: 0.31,
    },
  });
  await settled();
  settingsHttp.puts.length = 0;

  const store = useChatRuntimeStore.getState();
  store.setParams({ ...store.params, checkpoint: LLAMA });
  await settled();
  const written: Record<string, Record<string, unknown>> = {};
  for (const put of settingsHttp.puts) {
    Object.assign(
      written,
      (put.inferenceParamsByModel ?? {}) as Record<
        string,
        Record<string, unknown>
      >,
    );
  }
  assert.equal(written[QWEN]?.temperature, 0.31);
});

// One-level merging would drop the first of two one-field patches.
test("two edits to one model inside a debounce window both survive", async () => {
  useChatRuntimeStore.setState({
    settingsHydrated: false,
    rememberParamsPerModel: true,
    paramsByModel: {},
  });
  settingsHttp.settings = {
    inferenceParamsByModel: { [QWEN]: { temperature: 0.6, topP: 0.9 } },
  };
  await useChatRuntimeStore.getState().hydratePersistedSettings();
  useChatRuntimeStore.setState({
    params: {
      ...useChatRuntimeStore.getState().params,
      checkpoint: QWEN,
      temperature: 0.6,
      topP: 0.9,
    },
  });
  await settled();
  settingsHttp.puts.length = 0;

  const first = useChatRuntimeStore.getState();
  first.setParams({ ...first.params, temperature: 0.42 });
  const second = useChatRuntimeStore.getState();
  second.setParams({ ...second.params, topP: 0.11 });
  await settled();

  assert.deepEqual(
    settingsHttp.puts.map((put) => put.inferenceParamsByModel),
    [{ [QWEN]: { temperature: 0.42, topP: 0.11 } }],
    "one PUT carrying both edits, not the last one alone",
  );
});

// An external pick leaves the local model resident, so its context does not apply.
test("a resident GGUF context does not cap an external model", async () => {
  useChatRuntimeStore.setState({
    settingsHydrated: false,
    rememberParamsPerModel: true,
    loadedContextLength: 8192,
    paramsByModel: {},
    params: {
      ...useChatRuntimeStore.getState().params,
      checkpoint: EXTERNAL,
    },
  });
  settingsHttp.settings = { inferenceParams: { maxTokens: 32768 } };
  await useChatRuntimeStore.getState().hydratePersistedSettings();
  assert.equal(useChatRuntimeStore.getState().params.maxTokens, 32768);

  useChatRuntimeStore.setState({
    settingsHydrated: false,
    loadedContextLength: 8192,
    paramsByModel: {},
    params: { ...useChatRuntimeStore.getState().params, checkpoint: QWEN },
  });
  settingsHttp.settings = { inferenceParams: { maxTokens: 32768 } };
  await useChatRuntimeStore.getState().hydratePersistedSettings();
  assert.equal(useChatRuntimeStore.getState().params.maxTokens, 8192);
});
