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
const { DEFAULT_INFERENCE_PARAMS } = await import(
  "../src/features/chat/types/runtime.ts"
);

const QWEN = "unsloth/Qwen3.5-9B-GGUF";
const LLAMA = "unsloth/Llama-4-8B";
const EXTERNAL = "external::anthropic::claude-opus-5";
const TUNED = { temperature: 0.2, maxTokens: 4096, systemPrompt: "Be terse." };

const STATUS_CONTEXT_LENGTH = 131072;
/** A status response for a resident GGUF, recommending its own sampling. */
const STATUS = {
  inference: { temperature: 0.9, top_p: 0.5 },
  is_gguf: true,
  context_length: STATUS_CONTEXT_LENGTH,
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
      loadedContextLength: STATUS_CONTEXT_LENGTH,
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
  // over the tuning, so it was gone on the next launch too. Nothing is written
  // now, this browser having only read the entry, so the stored tuning stands.
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

// A stored entry can be partial: an older write, or a field that did not
// survive sanitising. It is kept as written and the replay lays it over what
// the load just published, which is where a gap belongs.
test("a partial stored entry is neither filled nor borrowed from", async () => {
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

  assert.deepEqual(
    useChatRuntimeStore.getState().paramsByModel[QWEN],
    { temperature: 0.15 },
    "stored as written, not grown with another model's settings",
  );

  // The load that follows publishes this model's own defaults, and the replay
  // lays the entry over them.
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

// The context belongs to the load config. A second copy recorded here is what
// would later replay over the context the backend actually loaded.
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

  // applyPerModelConfigToRuntime, staging the context of the model about to
  // load while the previous one is still current.
  const staging = useChatRuntimeStore.getState();
  staging.setParams({ ...staging.params, maxSeqLength: 32768 });
  assert.deepEqual(
    useChatRuntimeStore.getState().paramsByModel,
    {},
    "a context on its own is not an edit this remembers",
  );

  // The load lands and the checkpoint moves.
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

// A model loaded mid-flight has no entry, so the hydrated global set would hand
// it the previous model's sampling.
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

// The resident model is the one the saved global set describes, so its
// recommendation must not stand in front of those settings.
test("the resident model keeps the settings saved for it", async () => {
  settingsHttp.settings = {
    inferenceParams: { temperature: 0.2, systemPrompt: "tuned" },
  };
  settingsHttp.hold();
  useChatRuntimeStore.setState({
    // Nothing selected yet: a local checkpoint is not persisted, the first
    // status publishes it. The starting sampling differs from the status, so
    // the recommendation really does move it.
    params: {
      ...useChatRuntimeStore.getState().params,
      checkpoint: "",
      temperature: 0.5,
    },
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
    0.2,
    "the saved value, not the recommendation",
  );
  assert.equal(params.systemPrompt, "tuned");
});

// A restore after a hidden auto-load steps off the model that load put there.
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

// A visible switch still records it.
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

// A default equal to the outgoing model's value never moved, so it is not
// covered by the changed keys, but it is still this model's default.
test("a default equal to the previous model's value is still kept", async () => {
  settingsHttp.settings = {
    inferenceParams: { temperature: 0.2 },
    inferenceParamsByModel: {},
  };
  settingsHttp.hold();
  useChatRuntimeStore.setState({
    // Both models recommend 0.9, so the load moves nothing.
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

// A status that beat the settings response has already published the context
// the model loaded with, so the replay has to fit it too.
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
    // What the status published for the reduced context it loaded with.
    loadedContextLength: 8192,
    settingsHydrated: false,
  });
  const hydrating = useChatRuntimeStore.getState().hydratePersistedSettings();
  settingsHttp.release?.();
  await hydrating;

  assert.equal(useChatRuntimeStore.getState().params.maxTokens, 8192);
});

// A model's defaults are not settings it was used with: recording them makes
// the next defaults hook replay them over itself.
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
  // Wide enough for the window each call now records alongside the merge, and
  // still far short of the next fromModelDefaults site in either file.
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
      // The Qwen3 thinking-mode params applied after a load.
      "../src/features/chat/hooks/use-chat-model-runtime.ts",
      /setParams\(\{ \.\.\.store\.params, \.\.\.p \}, \{\s*fromModelDefaults: true,/,
    ],
  ];
  for (const [path, pattern] of sites) {
    const source = readFileSync(new URL(path, import.meta.url), "utf8");
    assert.match(source, pattern, path);
  }
});

// The user drags a slider while the GET is still out. The fence keeps the
// server's value off it, but the entry arriving for the model predates it.
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
  // Keys the user did not touch still come from the server.
  assert.equal(hydrated.paramsByModel[QWEN]?.systemPrompt, "stale");

  applyStatus(QWEN);
  assert.equal(
    useChatRuntimeStore.getState().params.temperature,
    0.33,
    "so a poll that re-applies defaults replays the edit, not the old value",
  );
});

// A safetensors reload at a smaller sequence length: the load sets the budget
// to that context and the memory would replay a larger one over it.
test("a remembered budget is capped by a non-GGUF load", () => {
  const runtime = readFileSync(
    new URL(
      "../src/features/chat/hooks/use-chat-model-runtime.ts",
      import.meta.url,
    ),
    "utf8",
  );
  // One cap for both sites: the load response and the Qwen3 thinking defaults.
  // The reported window leads, and the request stands in only for a backend that
  // sizes nothing -- a self-sizing one is sent the auto-size sentinel. Through the
  // floor, so a window below the control's own minimum cannot become the cap.
  assert.match(
    runtime,
    /const loadedContextCap = replayMaxTokensCap\(\s*loadedFields\.loadedContextLength \?\?\s*\(!loadResponse\.is_gguf && effectiveMaxSeqLength > 0\s*\? effectiveMaxSeqLength\s*: null\),\s*\);/,
  );
  assert.equal(
    runtime.match(/maxTokensCap: loadedContextCap/g)?.length,
    2,
    "the thinking-defaults replay is capped too",
  );

  const adapter = readFileSync(
    new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
    "utf8",
  );
  assert.match(
    adapter,
    /maxTokensCap: replayMaxTokensCap\(\s*candidate\.kind === "gguf"\s*\? loadedContextFields\(loadResp\)\.loadedContextLength\s*: loadedWindow,\s*\),/,
  );

  // Compare loads the same way: a pane with no context pin sends the sentinel, and
  // capping its budget at 0 would leave the pane asking for no output at all.
  const composer = readFileSync(
    new URL("../src/features/chat/shared-composer.tsx", import.meta.url),
    "utf8",
  );
  assert.match(
    composer,
    /maxTokensCap: replayMaxTokensCap\(\s*loadedContextFields\(resp\)\.loadedContextLength \?\?\s*\(!resp\.is_gguf && effectiveMaxSeqLength > 0/,
  );

  const status = readFileSync(
    new URL(
      "../src/features/chat/lib/apply-inference-status-to-store.ts",
      import.meta.url,
    ),
    "utf8",
  );
  // Reported for a safetensors load too, so the cap is not narrowed to GGUF.
  assert.match(status, /maxTokensCap: status\.context_length \?\? undefined,/);
});

// The clamp itself, through the store: the memory holds a budget from a larger
// context and the load reports a smaller one.
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

  // Without a cap the older, larger budget is what comes back.
  const uncapped = useChatRuntimeStore.getState();
  uncapped.setParams(
    { ...uncapped.params, maxTokens: 8192 },
    { fromModelDefaults: true },
  );
  assert.equal(useChatRuntimeStore.getState().params.maxTokens, 32768);
});

// The toggle is a mirrored scalar setting, so the write goes through
// setScalarSettingVersion rather than an explicit saveSettingsPatch beside it.
// Turning it off has to survive a reload, or the memory comes back on.
test("turning the memory off is persisted and hydrated back", async () => {
  useChatRuntimeStore.setState({
    settingsHydrated: true,
    rememberParamsPerModel: true,
  });
  settingsHttp.puts.length = 0;
  useChatRuntimeStore.getState().setRememberParamsPerModel(false);
  await settled();
  // The writer debounces and coalesces, so this is the patch the toggle joined.
  assert.equal(
    settingsHttp.puts.at(-1)?.rememberParamsPerModel,
    false,
    "the choice is written, not just held in the store",
  );

  // The next launch reads it back rather than falling to the default.
  useChatRuntimeStore.setState({
    settingsHydrated: false,
    rememberParamsPerModel: true,
  });
  settingsHttp.settings = { rememberParamsPerModel: false };
  await useChatRuntimeStore.getState().hydratePersistedSettings();
  assert.equal(useChatRuntimeStore.getState().rememberParamsPerModel, false);
});

// A safetensors load publishes its context through the cap, not through
// loadedContextLength, which a backend that sizes no window leaves null.
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

  // The status beats the settings response and reports the smaller context.
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

// The cap belongs to the model it was reported for: a switch away from it must
// not carry it onto the next one.
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
  // A different model takes over, with no context reported for it.
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

// The settings on screen got there by replay and a hidden load replays without
// persisting, so the global set can still be the previous model's.
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

  // A hidden restore: B's settings reach the screen, nothing is written.
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

// An install upgraded from before the memory has no entries at all, so the
// replay never runs and the cap that rides with it never applies.
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

// The server merges per key, so a full snapshot rewrites every field of a
// model's entry. A second tab that has only read an entry has nothing to say
// about it, and switching models is not an edit.
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

  // Switching back and forth, touching nothing.
  for (const checkpoint of [LLAMA, QWEN, LLAMA]) {
    const store = useChatRuntimeStore.getState();
    store.setParams({ ...store.params, checkpoint });
    assert.deepEqual(
      await perModelWrites(),
      [],
      "a switch reads the entries, it does not rewrite them",
    );
  }
  // The replay still happens, it is only the write that is withheld.
  assert.equal(useChatRuntimeStore.getState().params.temperature, 0.7);

  // An edit here is this browser's own, and is written -- but only the key it
  // moved. The server merges per key, so sending the rest would put this
  // browser's copy of the prompt over one the other tab has since changed.
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

  // And switching away from it writes nothing more: the edit already said it,
  // and the rest of the entry is not this browser's to restate.
  const leaving = useChatRuntimeStore.getState();
  leaving.setParams({ ...leaving.params, checkpoint: QWEN });
  assert.deepEqual(await perModelWrites(), []);
});

// The case the outgoing snapshot exists for: a model with no entry at all,
// switched away from without ever being edited, still has to be seeded.
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

// Two fields of one model changed inside the debounce window each send a
// one-field object, and one level of merging would drop the first.
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

// Picking an external model leaves the local one resident, so loadedContextLength
// goes on describing a model that has nothing to do with the pick.
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

  // A local checkpoint with the same resident context is still capped.
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
