// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A send that waits on a model load must not pick up the settings of the chat opened meanwhile.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import ts from "@typescript/typescript6";

import { installLocalStorageFake, readSrc } from "./helpers/kit.ts";
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
const { useChatRuntimeStore, codeToolsOn } = (await import(
  "../src/features/chat/stores/chat-runtime-store.ts"
)) as unknown as {
  useChatRuntimeStore: {
    getState: () => StoreState;
    setState: (patch: Record<string, unknown>) => void;
  };
  codeToolsOn: (state: StoreState) => boolean;
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
  // The model that just loaded is the one the run uses.
  assert.equal(run.params.checkpoint, "unsloth/new-model-GGUF");
  assert.equal(run.supportsTools, true);
  assert.equal(run.loadedContextLength, 32768);
  assert.equal(run.loadedIsGguf, true);
  // Not chat-scoped, so the live value stands.
  assert.equal(run.params.maxTokens, 2048);
  assert.equal(run.activeThreadId, "chat-b");
  // Neither input is mutated.
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

// The adapter's own post-wait selection, lifted from source: importing the adapter would drag in
// the stores, toasts and network layer. Dropping the snapshot from that block fails these tests.
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
  "resolvedThreadId",
  "withRunThreadScopedSettings",
  "mergeQueuedModelCapabilities",
  `let runtime;\n${ts.transpileModule(ADAPTER.slice(selectionStart, selectionEnd), {
    compilerOptions: { target: ts.ScriptTarget.ES2020 },
  }).outputText}\nreturn runtime;`,
) as (...args: unknown[]) => ReturnType<typeof chatState>;

function runAfterWait(
  live: ReturnType<typeof chatState>,
  atSend: ReturnType<typeof chatState>,
  resolvedThreadId: string | undefined,
) {
  return selectRunRuntime(
    { getState: () => live },
    null,
    null,
    atSend,
    resolvedThreadId,
    withRunThreadScopedSettings,
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
  const capture = ADAPTER.indexOf("      const runtimeAtSend = runtime;");
  const wait = ADAPTER.indexOf("await waitForModelReady(abortSignal);", capture);
  assert.ok(capture >= 0, "runtimeAtSend is captured");
  assert.ok(wait - capture < 400 && wait < selectionStart, "captured just before the wait");
});
