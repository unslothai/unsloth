// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A send that waits on a model load must not pick up the settings of the chat opened meanwhile.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import ts from "@typescript/typescript6";

import { installLocalStorageFake, readSrc } from "./helpers/kit.ts";
import { useModelReasoningEffortStore } from "../src/features/model-picker/components/model-selector/model-reasoning-effort.ts";
import {
  THREAD_SCOPED_PARAM_KEYS,
  THREAD_SCOPED_SETTING_KEYS,
  isThreadScopedParamKey,
  withRunThreadScopedSettings,
} from "../src/features/chat/utils/thread-scoped-settings.ts";

const { store: localStorageFake } = installLocalStorageFake();
localStorageFake.set("unsloth_chat_settings_imported_to_studio_db", "true");
register("./thread-sampling-resolver.mjs", import.meta.url);
await import("./helpers/store-stubs/settings-http.ts");
await import("./helpers/store-stubs/chat-history-storage.ts");

type StoreState = Record<string, unknown> & { params: { checkpoint: string } };
const {
  useChatRuntimeStore,
  codeToolsOn,
  withChatOwnedSettings,
  withLoadedModelOverrides,
  resolveToolsEnabledOnLoad,
} = (await import("../src/features/chat/stores/chat-runtime-store.ts")) as unknown as {
  useChatRuntimeStore: {
    getState: () => StoreState;
    setState: (patch: Record<string, unknown>) => void;
  };
  codeToolsOn: (state: StoreState) => boolean;
  withChatOwnedSettings: (state: StoreState) => StoreState;
  withLoadedModelOverrides: (run: StoreState, live: StoreState) => StoreState;
  resolveToolsEnabledOnLoad: (supportsTools: boolean) => Record<string, boolean>;
};

function chatState(overrides: Record<string, unknown>, params: Record<string, unknown>) {
  return {
    activeThreadId: "chat-a",
    permissionMode: "ask",
    confirmToolCalls: true,
    bypassPermissions: false,
    chatFullAccessPreviousMode: null,
    toolsEnabled: false,
    ragEnabled: false,
    ragSource: { type: "thread" },
    reasoningEffort: "low",
    supportsTools: false,
    loadedContextLength: 4096,
    loadedIsGguf: false,
    paramsByModel: {},
    ...overrides,
    params: {
      checkpoint: "",
      temperature: 0.2,
      topP: 0.9,
      systemPrompt: "chat A prompt",
      maxTokens: 512,
      ...params,
    },
  };
}

test("chat A keeps its permission mode, tools, prompt and sampling over chat B's", () => {
  const atSend = chatState({}, {});
  const afterSwitchAndLoad = chatState(
    {
      activeThreadId: "chat-b",
      permissionMode: "auto",
      toolsEnabled: true,
      ragEnabled: true,
      ragSource: { type: "project", projectId: "p" },
      reasoningEffort: "high",
      supportsTools: true,
      loadedContextLength: 32768,
      loadedIsGguf: true,
    },
    {
      checkpoint: "unsloth/new-model-GGUF",
      temperature: 1.1,
      topP: 0.5,
      systemPrompt: "chat B prompt",
      maxTokens: 2048,
    },
  );

  const run = withRunThreadScopedSettings(afterSwitchAndLoad, atSend);

  assert.equal(run.permissionMode, "ask");
  assert.equal(run.confirmToolCalls, true);
  assert.equal(run.toolsEnabled, false);
  assert.equal(run.ragEnabled, false);
  assert.deepEqual(run.ragSource, { type: "thread" });
  assert.equal(run.reasoningEffort, "low");
  assert.equal(run.params.systemPrompt, "chat A prompt");
  assert.equal(run.params.temperature, 0.2);
  assert.equal(run.params.topP, 0.9);
  assert.equal(run.params.checkpoint, "unsloth/new-model-GGUF");
  assert.equal(run.supportsTools, true);
  assert.equal(run.loadedContextLength, 32768);
  assert.equal(run.loadedIsGguf, true);
  assert.equal(run.params.maxTokens, 2048);
  assert.equal(run.activeThreadId, "chat-b");
  assert.equal(afterSwitchAndLoad.permissionMode, "auto");
  assert.equal(afterSwitchAndLoad.params.systemPrompt, "chat B prompt");
});

test("a chat-only full access grant keeps the mode it displaced", () => {
  const atSend = chatState(
    {
      permissionMode: "full",
      bypassPermissions: true,
      confirmToolCalls: false,
      chatFullAccessPreviousMode: "ask",
    },
    {},
  );
  const live = chatState({ activeThreadId: "chat-b", permissionMode: "off", confirmToolCalls: false }, {});

  const run = withRunThreadScopedSettings(live, atSend);

  assert.equal(run.permissionMode, "full");
  assert.equal(run.bypassPermissions, true);
  assert.equal(run.confirmToolCalls, false);
  assert.equal(run.chatFullAccessPreviousMode, "ask");
});

test("every thread-scoped key comes from the run's own chat", () => {
  const own: Record<string, unknown> = { params: {} };
  const live: Record<string, unknown> = { params: {} };
  for (const key of THREAD_SCOPED_SETTING_KEYS) {
    const target = isThreadScopedParamKey(key) ? "params" : null;
    const ownTarget = (target ? own.params : own) as Record<string, unknown>;
    const liveTarget = (target ? live.params : live) as Record<string, unknown>;
    ownTarget[key] = `own:${key}`;
    liveTarget[key] = `live:${key}`;
  }

  const run = withRunThreadScopedSettings(
    live as { params: object },
    own as { params: object },
  ) as Record<string, unknown> & { params: Record<string, unknown> };

  for (const key of THREAD_SCOPED_SETTING_KEYS) {
    const value = isThreadScopedParamKey(key) ? run.params[key] : run[key];
    assert.equal(value, `own:${key}`, key);
  }
  for (const key of THREAD_SCOPED_PARAM_KEYS) {
    assert.equal(Object.hasOwn(run, key), false, `${key} belongs under params`);
  }
});

test("chat A's Code-off under Full access holds after chat B picks Ask for approval", () => {
  const store = useChatRuntimeStore;
  store.setState({
    params: { ...store.getState().params, checkpoint: "unsloth/Qwen3-1.7B-GGUF" },
    supportsTools: true,
  });
  const call = (name: string, value: unknown) =>
    (store.getState()[name] as (value: unknown) => void)(value);
  call("setPermissionMode", "full");
  call("setCodeToolsEnabled", false);
  const atSend = store.getState();
  assert.equal(codeToolsOn(atSend), false, "chat A switched Code off");

  // Full access outlives the switch; picking Ask in chat B clears the decline.
  call("setPermissionMode", "ask");
  const live = store.getState();
  assert.equal(live.codeToolsDeclinedUnderFullAccess, false);

  const run = withRunThreadScopedSettings(live, atSend);
  assert.equal(run.permissionMode, "full");
  assert.equal(codeToolsOn(run), false);
});

// The adapter's post-wait block, lifted from source: importing the adapter drags in stores and network.
const ADAPTER = readSrc("features/chat/api/chat-adapter.ts");
const selectionStart = ADAPTER.indexOf(
  "      const liveRuntime = useChatRuntimeStore.getState();",
);
const selectionEnd = ADAPTER.indexOf("      runtime = withChatFullAccess(", selectionStart);
assert.ok(selectionStart >= 0 && selectionEnd > selectionStart, "post-wait selection moved");
const selectRunRuntime = new Function(
  "useChatRuntimeStore",
  "queuedRunSettings",
  "queuedEmptyModelRuntime",
  "runtimeAtSend",
  "chatOwnedAtSend",
  "resolvedThreadId",
  "withRunThreadScopedSettings",
  "withLoadedModelOverrides",
  "mergeQueuedModelCapabilities",
  `let runtime;\n${ts.transpileModule(ADAPTER.slice(selectionStart, selectionEnd), {
    compilerOptions: { target: ts.ScriptTarget.ES2020 },
  }).outputText}\nreturn runtime;`,
) as (...args: unknown[]) => ReturnType<typeof chatState>;

function runAfterWait(
  live: ReturnType<typeof chatState>,
  atSend: ReturnType<typeof chatState>,
  resolvedThreadId: string | undefined,
  overrides: (run: never, live: never) => unknown = (run) => run,
  chatOwned: unknown = atSend,
) {
  return selectRunRuntime(
    { getState: () => live },
    null,
    null,
    atSend,
    chatOwned,
    resolvedThreadId,
    withRunThreadScopedSettings,
    overrides,
    () => {
      throw new Error("a direct send has no queued model capabilities");
    },
  );
}

const loadedB = (overrides: Record<string, unknown> = {}) =>
  chatState(
    { activeThreadId: "chat-b", permissionMode: "auto", supportsTools: true, ...overrides },
    { checkpoint: "unsloth/new-model-GGUF", systemPrompt: "chat B prompt", temperature: 1.1 },
  );

test("the adapter keeps the sent chat's settings when another chat is on screen", () => {
  const run = runAfterWait(loadedB(), chatState({}, {}), "chat-a");
  assert.equal(run.permissionMode, "ask");
  assert.equal(run.params.systemPrompt, "chat A prompt");
  assert.equal(run.params.temperature, 0.2);
  assert.equal(run.params.checkpoint, "unsloth/new-model-GGUF");
  assert.equal(run.supportsTools, true);
});

test("the adapter reads the live store while the sent chat is still on screen", () => {
  const live = loadedB({ activeThreadId: "chat-a" });
  assert.equal(runAfterWait(live, chatState({}, {}), "chat-a"), live);
  // A first send: no id on screen at send, then the new chat's own id.
  const firstSend = loadedB({ activeThreadId: "chat-new" });
  assert.equal(
    runAfterWait(firstSend, chatState({ activeThreadId: null }, {}), "chat-new"),
    firstSend,
  );
});

test("the adapter keeps the sent chat's settings while the shown chat's settings load", () => {
  const live = loadedB({ activeThreadId: "chat-a", threadScopedSettingsPending: true });
  const run = runAfterWait(live, chatState({}, {}), "chat-a");
  assert.equal(run.permissionMode, "ask");
  assert.equal(run.params.systemPrompt, "chat A prompt");
});

test("the adapter takes its send-time settings before the model wait", () => {
  const capture = ADAPTER.indexOf(
    "      const runtimeAtSend = runtime;\n      const chatOwnedAtSend = withChatOwnedSettings(runtime);",
  );
  const wait = ADAPTER.indexOf("await waitForModelReady(abortSignal);", capture);
  assert.ok(capture >= 0, "runtimeAtSend is captured");
  assert.ok(wait - capture < 400 && wait < selectionStart, "captured just before the wait");
});

test("chat A's tools follow its choice, not the clamp the outgoing model left", () => {
  const store = useChatRuntimeStore;
  const call = (name: string, value: unknown) =>
    (store.getState()[name] as (value: unknown) => void)(value);
  store.setState({ params: { ...store.getState().params, checkpoint: "tools-model" }, supportsTools: true });
  call("setPermissionMode", "ask");
  call("setToolsEnabled", true);
  call("setCodeToolsEnabled", true);
  store.setState({
    params: { ...store.getState().params, checkpoint: "plain-model" },
    supportsTools: false,
    ...resolveToolsEnabledOnLoad(false),
  });
  store.setState({ modelLoading: true });
  assert.equal(store.getState().toolsEnabled, false, "the outgoing model clamps the pill");
  const atSend = withChatOwnedSettings(store.getState());

  const loaded = {
    ...store.getState(),
    activeThreadId: "chat-b",
    modelLoading: false,
    supportsTools: true,
    params: { ...store.getState().params, checkpoint: "unsloth/new-model-GGUF" },
  } as unknown as ReturnType<typeof chatState>;
  const run = runAfterWait(loaded, atSend as never, "chat-a", withLoadedModelOverrides as never) as unknown as StoreState;
  assert.equal(run.toolsEnabled, true);
  assert.equal(codeToolsOn(run), true);

  const plain = { ...loaded, supportsTools: false } as typeof loaded;
  const offRun = runAfterWait(plain, atSend as never, "chat-a", withLoadedModelOverrides as never) as unknown as StoreState;
  assert.equal(offRun.toolsEnabled, false);
  assert.equal(offRun.codeToolsEnabled, false);
  store.setState({ modelLoading: false });
});

test("chat A runs at its own effort, not the outgoing model's pin", () => {
  const store = useChatRuntimeStore;
  store.setState({ settingsHydrated: true, modelLoading: false });
  (store.getState().applyThreadScopedSettings as (id: string, s: object) => void)("chat-a", {
    reasoningEffort: "low",
  });
  useModelReasoningEffortStore.setState({
    effortByModel: { "pinned-model": "high", "other-pinned-model": "medium" },
  });
  const reasoning = {
    supportsReasoning: true,
    reasoningStyle: "reasoning_effort",
    reasoningEffortLevels: ["low", "medium", "high"],
  };
  store.setState({
    ...reasoning,
    params: { ...store.getState().params, checkpoint: "pinned-model" },
    reasoningEffort: "high",
    modelLoading: true,
  });
  const atSend = store.getState();
  const chatOwned = withChatOwnedSettings(atSend);
  assert.equal(chatOwned.reasoningEffort, "low");

  const loaded = (checkpoint: string, reasoningEffort: string) =>
    ({
      ...store.getState(),
      ...reasoning,
      activeThreadId: "chat-b",
      modelLoading: false,
      reasoningEffort,
      params: { ...store.getState().params, checkpoint },
    }) as unknown as ReturnType<typeof chatState>;
  const run = (live: ReturnType<typeof chatState>) =>
    runAfterWait(live, atSend as never, "chat-a", withLoadedModelOverrides as never, chatOwned) as unknown as StoreState;
  assert.equal(run(loaded("unsloth/local-model", "high")).reasoningEffort, "low");
  assert.equal(run(loaded("other-pinned-model", "medium")).reasoningEffort, "medium");
  assert.equal(run(loaded("pinned-model", "low")).reasoningEffort, "high");
  store.setState({ modelLoading: false });
});

test("chat A's Think on a new model is what that model's load set", () => {
  const atSend = chatState({ reasoningEnabled: true }, {});
  const loaded = (reasoningEnabled: boolean) =>
    loadedB({ activeGgufVariant: "Q4_K_M", supportsReasoning: true, reasoningEnabled });
  const think = (reasoningEnabled: boolean) =>
    (runAfterWait(loaded(reasoningEnabled), atSend, "chat-a", withLoadedModelOverrides as never) as unknown as StoreState)
      .reasoningEnabled;
  // A Think pin, a model that cannot stop thinking, or a Think-off default: the load decided.
  assert.equal(think(true), true);
  assert.equal(think(false), false);
});

test("a load that leaves the model in place keeps chat A's effective values", () => {
  // Kimi: Think on suppressed Search without persisting it; the chat's record still has Search on.
  const model = { params: { checkpoint: "kimi::kimi-k2.6" }, activeGgufVariant: null };
  // A rolled-back load restores the variant as null where the send may have had none at all.
  const atSend = chatState(
    { reasoningEnabled: true, toolsEnabled: false, ...model, activeGgufVariant: undefined },
    model.params,
  );
  const chatOwned = chatState({ reasoningEnabled: true, toolsEnabled: true, ...model }, model.params);
  const live = chatState(
    { activeThreadId: "chat-b", reasoningEnabled: false, toolsEnabled: true, permissionMode: "auto", ...model },
    { ...model.params, systemPrompt: "chat B prompt" },
  );
  const run = runAfterWait(live, atSend, "chat-a", withLoadedModelOverrides as never, chatOwned) as unknown as StoreState;
  assert.equal(run.toolsEnabled, false);
  assert.equal(run.reasoningEnabled, true);
  assert.equal(run.permissionMode, "ask");
  assert.equal((run.params as unknown as { systemPrompt: string }).systemPrompt, "chat A prompt");
});

test("chat A's unowned sampling follows the new model's load; its own stays", () => {
  const store = useChatRuntimeStore;
  const apply = store.getState().applyThreadScopedSettings as (id: string, s: object) => void;
  const send = (thread: string, stored: object, recommended = 0.31) => {
    store.setState({
      settingsHydrated: true,
      modelLoading: true,
      activeThreadId: thread,
      params: { ...store.getState().params, checkpoint: "old-model", temperature: 0.6, topP: 0.95 },
    });
    apply(thread, stored);
    const atSend = store.getState();
    const chatOwned = withChatOwnedSettings(atSend);
    apply("chat-b", { temperature: 1.1 });
    store.setState({ activeThreadId: "chat-b", modelLoading: false });
    (store.getState().setParams as (p: object, o: object) => void)(
      { ...store.getState().params, checkpoint: "new-model", temperature: recommended, topP: 0.41 },
      { fromModelDefaults: true },
    );
    const live = store.getState() as unknown as ReturnType<typeof chatState>;
    assert.equal(live.params.temperature, 1.1, "chat B's own value is on screen");
    const run = runAfterWait(live, atSend as never, thread, withLoadedModelOverrides as never, chatOwned);
    store.setState({ modelLoading: false });
    return run.params;
  };
  // A legacy row stores other settings but no sampling (thread-scoped-sampling-compat C1b).
  const legacy = send("legacy-a", { permissionMode: "ask" });
  assert.equal(legacy.temperature, 0.31);
  assert.equal(legacy.topP, 0.41);
  // Remembered per-model values are not the load's when model memory is off.
  store.setState({ rememberParamsPerModel: false, paramsByModel: { "new-model": { temperature: 1.5 } } });
  assert.equal(send("off-a", { permissionMode: "ask" }).temperature, 0.31);
  store.setState({ rememberParamsPerModel: true, paramsByModel: {} });
  // Chat B already holds the recommendation, so the load moves no stored default.
  assert.equal(send("same-a", { permissionMode: "ask" }, 1.1).temperature, 1.1);
  const pinned = send("pinned-a", { permissionMode: "ask", temperature: 0.2 });
  assert.equal(pinned.temperature, 0.2);
  assert.equal(pinned.topP, 0.41);
});
