// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import type { ThreadMessage } from "@assistant-ui/react";
import ts from "@typescript/typescript6";

import { isExternalModelId } from "../src/features/chat/external-providers.ts";
import { contextUsageReasoningKey } from "../src/features/chat/utils/context-usage-reasoning.ts";
import * as savedUsage from "../src/features/chat/utils/saved-context-usage.ts";

const source = ts.createSourceFile(
  "runtime-provider.tsx",
  readFileSync(new URL("../src/features/chat/runtime-provider.tsx", import.meta.url), "utf8"),
  ts.ScriptTarget.ES2022,
  true,
  ts.ScriptKind.TSX,
);

const refreshSource = ts.createSourceFile(
  "refresh-context-usage.ts",
  readFileSync(new URL("../src/features/chat/utils/refresh-context-usage.ts", import.meta.url), "utf8"),
  ts.ScriptTarget.ES2022,
  true,
  ts.ScriptKind.TS,
);

const browserToolsSource = ts.createSourceFile(
  "tools.ts",
  readFileSync(new URL("../src/features/browser/agent/tools.ts", import.meta.url), "utf8"),
  ts.ScriptTarget.ES2022,
  true,
  ts.ScriptKind.TS,
);

const storeSource = ts.createSourceFile(
  "chat-runtime-store.ts",
  readFileSync(new URL("../src/features/chat/stores/chat-runtime-store.ts", import.meta.url), "utf8"),
  ts.ScriptTarget.ES2022,
  true,
  ts.ScriptKind.TS,
);

function runtimeFunction(name: string, dependencies: Record<string, unknown>, sourceFile = source) {
  const declaration = sourceFile.statements.find(
    (statement): statement is ts.FunctionDeclaration =>
      ts.isFunctionDeclaration(statement) && statement.name?.text === name,
  );
  assert.ok(declaration, `${name} exists`);
  const { outputText } = ts.transpileModule(declaration.getText(sourceFile).replace(/^export /, ""), {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
  });
  return new Function(...Object.keys(dependencies), `${outputText}\nreturn ${name};`)(
    ...Object.values(dependencies),
  ) as (props: { enabled: boolean }) => null;
}

function usage(totalTokens: number, reasoningKey?: string): savedUsage.BranchContextUsage {
  return { promptTokens: totalTokens, completionTokens: 0, totalTokens, cachedTokens: 0, reasoningKey };
}

function reply(count: savedUsage.BranchContextUsage): ThreadMessage {
  return {
    id: "reply",
    role: "assistant",
    createdAt: new Date(0),
    content: [{ type: "text", text: "Saved reply" }],
    status: { type: "complete", reason: "stop" },
    metadata: {
      custom: { contextUsage: { ...count, modelId: "local-model" } },
      steps: [],
      unstable_annotations: [],
      unstable_data: [],
      unstable_state: null,
    },
  };
}

function harness(name = "ThreadContextUsageRecount") {
  const state = {
    activeThreadId: "thread-a" as string | null,
    params: { checkpoint: "local-model", systemPrompt: "Original instructions", temperature: 0.6 },
    loadedChatTemplateOverride: "original-template",
    loadedContextLength: 4096 as number | null,
    loadedIsGguf: true,
    loadedIsMlx: false,
    modelLoading: false,
    threadScopedSettingsPending: false,
    queuedSettingsEpoch: 0,
    toolsEnabled: false,
    supportsTools: true,
    supportsBuiltinImageGeneration: false,
    imageToolsEnabled: false,
    localImageModelReady: false,
    localVideoModelReady: false,
    reasoningKey: undefined as string | undefined,
    runningByThreadId: {} as Record<string, boolean>,
    contextUsage: usage(42, contextUsageReasoningKey({})) as savedUsage.BranchContextUsage | null,
    setContextUsage: (next: savedUsage.BranchContextUsage | null) => {
      state.contextUsage = next;
      writes.push(next);
    },
    setThreadContextUsage: () => {},
  };
  const world = {
    mainThreadId: "thread-a",
    thread: { isLoading: false, messages: [reply(state.contextUsage!)] },
    recounts: [] as unknown[],
    image: false,
    audio: false,
    video: false,
    browser: {
      enabled: false,
      historyAccess: "ask" as "ask" | "allow" | "never",
      scriptAccess: "ask" as "ask" | "never",
    },
    browserToolSelections: [] as unknown[],
  };
  const writes: Array<savedUsage.BranchContextUsage | null> = [];
  const refs: Array<{ current: unknown }> = [];
  const previousDeps: Array<readonly unknown[]> = [];
  let pendingEffects: Array<() => unknown> = [];
  let hookIndex = 0;
  let branchReader: (() => readonly ThreadMessage[] | null) | null = null;
  const aui = {
    thread: () => ({ getState: () => world.thread }),
    threads: () => ({ getState: () => ({ mainThreadId: world.mainThreadId }) }),
  };
  const useChatRuntimeStore = Object.assign(
    (selector: (snapshot: typeof state) => unknown) => selector(state),
    { getState: () => state },
  );
  const useBrowserAgentStore = Object.assign(
    (selector: (snapshot: typeof world.browser) => unknown) => {
      const selected = selector(world.browser);
      world.browserToolSelections.push(selected);
      return selected;
    },
    { getState: () => world.browser },
  );
  const browserAgentToolsForTurn = runtimeFunction("browserAgentToolsForTurn", {
    useBrowserAgentStore,
    browserPanelAvailable: () => true,
    isTauri: false,
    BROWSER_AGENT_TOOLS: ["browser_navigate"],
    BROWSER_SCREENSHOT_TOOL: "browser_screenshot",
    BROWSER_HISTORY_TOOL: "browser_history",
    BROWSER_SCRIPT_TOOL: "browser_evaluate",
  }, browserToolsSource);
  const component = runtimeFunction(name, {
    ...savedUsage,
    isExternalModelId,
    currentContextUsageReasoningKey: (snapshot = state) => snapshot.reasoningKey,
    currentContextUsageSettingsKey: runtimeFunction("currentContextUsageSettingsKey", {
      localMediaToolsForTurn: runtimeFunction("localMediaToolsForTurn", {
        localImageTool: runtimeFunction("localImageTool", {}, storeSource),
      }, storeSource),
    }, refreshSource),
    useChatRuntimeStore,
    useBrowserAgentStore,
    browserAgentToolsForTurn,
    useAui: () => aui,
    useAuiState: (selector: (snapshot: { thread: typeof world.thread }) => unknown) => selector(world),
    usePromptQueueUI: (selector: (snapshot: { byThreadId: Record<string, never> }) => unknown) =>
      selector({ byThreadId: {} }),
    useRef: (initial: unknown) => refs[hookIndex++] ??= { current: initial },
    useEffect: (effect: () => unknown, dependencies: readonly unknown[]) => {
      const index = hookIndex++;
      const previous = previousDeps[index];
      previousDeps[index] = dependencies;
      if (!previous || dependencies.some((value, i) => !Object.is(value, previous[i]))) {
        pendingEffects.push(effect);
      }
    },
    refreshContextUsage: (options: unknown) => {
      world.recounts.push(options);
      return Promise.resolve();
    },
    setActiveBranchReader: (reader: typeof branchReader) => { branchReader = reader; },
    messagesContainImage: () => world.image,
    findLatestUserAudioBase64: () => world.audio ? "audio" : null,
    findLatestUserVideoBase64: () => world.video ? "video" : null,
  });
  const render = () => {
    hookIndex = 0;
    pendingEffects = [];
    component({ enabled: true });
    for (const effect of pendingEffects) effect();
  };
  return { state, world, writes, render, readBranch: () => branchReader?.() };
}

test("pairing a non-reasoning chat preserves its recounted usage", () => {
  const h = harness();
  h.world.thread.messages = [reply(usage(114))];
  h.state.contextUsage = usage(101, contextUsageReasoningKey({}));
  const recounted = h.state.contextUsage;
  h.state.threadScopedSettingsPending = true;
  h.render();
  h.state.threadScopedSettingsPending = false;
  h.render();
  assert.equal(h.state.contextUsage, recounted);
  assert.deepEqual(h.writes, []);
  assert.deepEqual(h.world.recounts, []);
});

test("pairing the first send for a non-reasoning model does not clear the exact prompt count", () => {
  const h = harness();
  h.world.thread.messages = [];
  const counted = h.state.contextUsage;
  h.state.threadScopedSettingsPending = true;
  h.render();
  h.state.threadScopedSettingsPending = false;
  h.render();
  assert.equal(h.state.contextUsage, counted);
  assert.deepEqual(h.writes, []);
  assert.deepEqual(h.world.recounts, []);
});

test("an effective reasoning change still estimates and recounts the loaded branch", () => {
  const h = harness();
  h.state.reasoningKey = contextUsageReasoningKey({ enable_thinking: true });
  h.state.contextUsage = usage(42, h.state.reasoningKey);
  h.world.thread.messages = [reply(h.state.contextUsage)];
  h.render();
  h.state.reasoningKey = contextUsageReasoningKey({ enable_thinking: false });
  h.render();
  assert.equal(h.state.contextUsage?.estimated, true);
  assert.equal(h.state.contextUsage?.totalTokens, 42);
  assert.equal(h.world.recounts.length, 1);
});

for (const unavailable of [null, "image", "audio", "video", "unloaded", "compacted", "unsupported"] as const) {
  test(`returning to the reply's reasoning mode restores its exact count (${unavailable ?? "text"})`, () => {
    const h = harness();
    const enabled = contextUsageReasoningKey({ enable_thinking: true });
    const disabled = contextUsageReasoningKey({ enable_thinking: false });
    h.state.reasoningKey = enabled;
    const saved = { ...usage(643, enabled), estimatedTokens: 6 };
    h.state.contextUsage = saved;
    h.world.thread.messages = [reply(saved)];
    if (unavailable === "image" || unavailable === "audio" || unavailable === "video") {
      h.world[unavailable] = true;
    } else if (unavailable === "unloaded") {
      h.state.loadedContextLength = null;
    } else if (unavailable === "compacted") {
      h.state.loadedIsGguf = false;
      h.state.loadedIsMlx = true;
      h.world.thread.messages[0].metadata.custom.contextTruncation = { summary: "Earlier turns" };
    } else if (unavailable === "unsupported") {
      h.state.loadedIsGguf = false;
    }
    h.render();
    h.state.reasoningKey = disabled;
    h.state.queuedSettingsEpoch += 1;
    h.render();
    assert.equal(h.state.contextUsage?.estimated, true);
    assert.ok((h.state.contextUsage?.totalTokens ?? 0) > 600);
    assert.equal(h.world.recounts.length, unavailable === null ? 1 : 0);
    h.state.reasoningKey = enabled;
    h.state.queuedSettingsEpoch += 1;
    h.render();
    assert.equal(h.state.contextUsage, h.world.thread.messages[0].metadata.custom.contextUsage);
    assert.equal(h.world.recounts.length, unavailable === null ? 1 : 0);
  });
}

test("changing reasoning after opening a legacy chat estimates and recounts its saved usage", () => {
  const h = harness();
  h.state.reasoningKey = contextUsageReasoningKey({ enable_thinking: true });
  h.state.contextUsage = usage(84);
  h.world.thread.messages = [reply(h.state.contextUsage)];
  h.render();
  h.state.reasoningKey = contextUsageReasoningKey({ enable_thinking: false });
  h.render();
  assert.equal(h.state.contextUsage?.estimated, true);
  assert.equal(h.world.recounts.length, 1);
});

for (const setting of ["systemPrompt", "chatTemplate", "tools"] as const) {
  test(`returning to a saved reasoning mode recounts after ${setting} changes`, () => {
    const h = harness();
    const enabled = contextUsageReasoningKey({ enable_thinking: true });
    const disabled = contextUsageReasoningKey({ enable_thinking: false });
    h.state.reasoningKey = enabled;
    h.state.contextUsage = usage(84, enabled);
    h.world.thread.messages = [reply(h.state.contextUsage)];
    h.render();
    h.state.reasoningKey = disabled;
    h.state.queuedSettingsEpoch += 1;
    h.render();
    if (setting === "systemPrompt") {
      h.state.params.systemPrompt = "Updated instructions";
    } else if (setting === "chatTemplate") {
      h.state.loadedChatTemplateOverride = "updated-template";
    } else {
      h.state.toolsEnabled = true;
      h.state.queuedSettingsEpoch += 1;
    }
    h.state.contextUsage = usage(1990, disabled);
    h.render();
    const recounts = h.world.recounts.length;
    h.state.reasoningKey = enabled;
    h.state.queuedSettingsEpoch += 1;
    h.render();
    assert.equal(h.state.contextUsage?.estimated, true);
    assert.equal(h.state.contextUsage?.totalTokens, 1990);
    assert.equal(h.world.recounts.length, recounts + 1);
  });
}

for (const setting of ["enabled", "historyAccess", "scriptAccess"] as const) {
  test(`browser ${setting} changes are recounted only after thinking changes`, () => {
    const h = harness();
    const enabled = contextUsageReasoningKey({ enable_thinking: true });
    const disabled = contextUsageReasoningKey({ enable_thinking: false });
    h.state.reasoningKey = enabled;
    h.state.contextUsage = usage(84, enabled);
    h.world.thread.messages = [reply(h.state.contextUsage)];
    h.world.browser.enabled = setting !== "enabled";
    h.render();
    const initialTools = h.world.browserToolSelections.at(-1);
    assert.equal(typeof initialTools, "string");
    const saved = h.state.contextUsage;
    if (setting === "enabled") {
      h.world.browser.enabled = true;
    } else {
      h.world.browser[setting] = "never";
    }
    h.render();
    assert.notEqual(h.world.browserToolSelections.at(-1), initialTools);
    assert.equal(h.state.contextUsage, saved);
    assert.deepEqual(h.world.recounts, []);
    h.state.reasoningKey = disabled;
    h.render();
    assert.equal(h.state.contextUsage?.estimated, true);
    assert.equal(h.world.recounts.length, 1);
    h.state.contextUsage = usage(1990, disabled);
    h.state.reasoningKey = enabled;
    h.render();
    assert.equal(h.state.contextUsage?.estimated, true);
    assert.equal(h.state.contextUsage?.totalTokens, 1990);
    assert.equal(h.world.recounts.length, 2);
  });
}

for (const browserEnabled of [false, true]) {
  test(`unchanged browser tools preserve saved usage when browser is ${browserEnabled ? "enabled" : "disabled"}`, () => {
    const h = harness();
    const enabled = contextUsageReasoningKey({ enable_thinking: true });
    const disabled = contextUsageReasoningKey({ enable_thinking: false });
    h.state.reasoningKey = enabled;
    h.state.contextUsage = usage(84, enabled);
    h.world.thread.messages = [reply(h.state.contextUsage)];
    h.world.browser.enabled = browserEnabled;
    h.render();
    const initialTools = h.world.browserToolSelections.at(-1);
    assert.equal(typeof initialTools, "string");
    h.world.browser.historyAccess = browserEnabled ? "allow" : "never";
    if (!browserEnabled) h.world.browser.scriptAccess = "never";
    h.render();
    assert.equal(h.world.browserToolSelections.at(-1), initialTools);
    assert.deepEqual(h.world.recounts, []);
    h.state.reasoningKey = disabled;
    h.render();
    h.state.contextUsage = usage(1990, disabled);
    h.state.reasoningKey = enabled;
    h.render();
    assert.equal(h.state.contextUsage, h.world.thread.messages[0].metadata.custom.contextUsage);
    assert.equal(h.world.recounts.length, 1);
  });
}

for (const setting of ["imageToolsEnabled", "localImageModelReady", "localVideoModelReady"] as const) {
  test(`local media ${setting} changes invalidate saved usage when thinking changes`, () => {
    const h = harness();
    const enabled = contextUsageReasoningKey({ enable_thinking: true });
    const disabled = contextUsageReasoningKey({ enable_thinking: false });
    h.state.reasoningKey = enabled;
    h.state.contextUsage = usage(84, enabled);
    h.world.thread.messages = [reply(h.state.contextUsage)];
    h.state.imageToolsEnabled = true;
    h.state.localImageModelReady = true;
    h.render();
    const saved = h.state.contextUsage;
    h.state[setting] = !h.state[setting];
    h.render();
    assert.equal(h.state.contextUsage, saved);
    assert.deepEqual(h.world.recounts, []);
    h.state.reasoningKey = disabled;
    h.render();
    assert.equal(h.state.contextUsage?.estimated, true);
    assert.equal(h.world.recounts.length, 1);
    h.state.contextUsage = usage(1990, disabled);
    h.state.reasoningKey = enabled;
    h.render();
    assert.equal(h.state.contextUsage?.estimated, true);
    assert.equal(h.state.contextUsage?.totalTokens, 1990);
    assert.equal(h.world.recounts.length, 2);
  });
}

for (const unavailable of ["disabled", "unsupported", "hosted"] as const) {
  test(`media readiness preserves saved usage when local tools are ${unavailable}`, () => {
    const h = harness();
    const enabled = contextUsageReasoningKey({ enable_thinking: true });
    const disabled = contextUsageReasoningKey({ enable_thinking: false });
    h.state.reasoningKey = enabled;
    h.state.contextUsage = usage(84, enabled);
    h.world.thread.messages = [reply(h.state.contextUsage)];
    h.state.imageToolsEnabled = unavailable !== "disabled";
    h.state.supportsTools = unavailable !== "unsupported";
    h.state.supportsBuiltinImageGeneration = unavailable === "hosted";
    h.render();
    h.state.localImageModelReady = true;
    h.state.localVideoModelReady = true;
    h.render();
    assert.deepEqual(h.world.recounts, []);
    h.state.reasoningKey = disabled;
    h.render();
    h.state.contextUsage = usage(1990, disabled);
    h.state.reasoningKey = enabled;
    h.render();
    assert.equal(h.state.contextUsage, h.world.thread.messages[0].metadata.custom.contextUsage);
    assert.equal(h.world.recounts.length, 1);
  });
}

test("a reply generated after prompt changes can be restored without another count", () => {
  const h = harness();
  const enabled = contextUsageReasoningKey({ enable_thinking: true });
  const disabled = contextUsageReasoningKey({ enable_thinking: false });
  h.state.reasoningKey = enabled;
  h.state.contextUsage = usage(84, enabled);
  h.world.thread.messages = [reply(h.state.contextUsage)];
  h.render();
  h.state.params.systemPrompt = "Updated instructions";
  h.state.queuedSettingsEpoch += 1;
  h.render();
  const next = { ...usage(1990, enabled), estimatedTokens: 6 };
  h.world.thread.messages = [reply(next)];
  h.state.contextUsage = next;
  h.render();
  h.state.reasoningKey = disabled;
  h.state.queuedSettingsEpoch += 1;
  h.render();
  const recounts = h.world.recounts.length;
  h.state.reasoningKey = enabled;
  h.state.queuedSettingsEpoch += 1;
  h.render();
  assert.equal(h.state.contextUsage, h.world.thread.messages[0].metadata.custom.contextUsage);
  assert.equal(h.world.recounts.length, recounts);
});

test("switching to a reply priced before a prompt change recounts it", () => {
  const h = harness();
  h.state.contextUsage = usage(84, contextUsageReasoningKey({}));
  h.world.thread.messages = [reply(h.state.contextUsage)];
  h.render();
  h.state.params.systemPrompt = "Updated instructions";
  h.state.queuedSettingsEpoch += 1;
  h.render();
  h.state.contextUsage = usage(1990, contextUsageReasoningKey({}));
  const recounts = h.world.recounts.length;
  h.world.thread.messages = [{ ...reply(usage(84, contextUsageReasoningKey({}))), id: "sibling" }];
  h.render();
  assert.equal(h.state.contextUsage?.estimated, true);
  assert.equal(h.world.recounts.length, recounts + 1);
});

test("batched tool and reasoning changes cannot restore the old prompt count", () => {
  const h = harness();
  const enabled = contextUsageReasoningKey({ enable_thinking: true });
  const disabled = contextUsageReasoningKey({ enable_thinking: false });
  h.state.reasoningKey = enabled;
  h.state.contextUsage = usage(84, enabled);
  h.world.thread.messages = [reply(h.state.contextUsage)];
  h.render();
  h.state.reasoningKey = disabled;
  h.state.toolsEnabled = true;
  h.state.queuedSettingsEpoch += 2;
  h.render();
  h.state.contextUsage = usage(1990, disabled);
  const recounts = h.world.recounts.length;
  h.state.reasoningKey = enabled;
  h.state.queuedSettingsEpoch += 1;
  h.render();
  assert.equal(h.state.contextUsage?.estimated, true);
  assert.equal(h.state.contextUsage?.totalTokens, 1990);
  assert.equal(h.world.recounts.length, recounts + 1);
});

test("thinking toggles that also update sampling and preserve settings retain the saved count", () => {
  const h = harness();
  const enabled = contextUsageReasoningKey({ enable_thinking: true, preserve_thinking: false });
  const disabled = contextUsageReasoningKey({ enable_thinking: false, preserve_thinking: false });
  h.state.reasoningKey = enabled;
  h.state.contextUsage = usage(643, enabled);
  h.world.thread.messages = [reply(h.state.contextUsage)];
  h.world.image = true;
  h.render();
  h.state.reasoningKey = disabled;
  h.state.params.temperature = 0.7;
  h.state.queuedSettingsEpoch += 3;
  h.render();
  assert.equal(h.state.contextUsage?.estimated, true);
  h.state.reasoningKey = enabled;
  h.state.params.temperature = 0.6;
  h.state.queuedSettingsEpoch += 2;
  h.render();
  assert.equal(h.state.contextUsage, h.world.thread.messages[0].metadata.custom.contextUsage);
  assert.deepEqual(h.world.recounts, []);
});

test("deleting an attachment invalidates exact usage when only the branch revision changes", () => {
  const h = harness();
  const user: ThreadMessage = {
    id: "user",
    role: "user",
    createdAt: new Date(0),
    content: [
      { type: "text", text: "Read my file" },
      { type: "image", image: "data:image/png;base64,aW1hZ2U=" },
    ],
    attachments: [],
    metadata: { custom: {} },
  };
  const messages = [user, reply(usage(643))];
  const saved = {
    ...usage(643),
    ...savedUsage.contextUsageMarkers(savedUsage.threadBranchRecords(messages)),
  };
  h.world.thread.messages = [user, reply(saved)];
  h.world.image = true;
  h.state.contextUsage = saved;
  h.render();
  assert.deepEqual(h.world.recounts, []);
  const tip = h.world.thread.messages.at(-1);
  const length = h.world.thread.messages.length;
  h.world.thread.messages = [
    {
      ...user,
      content: user.content.slice(0, 1),
      metadata: { custom: { revision: 1 } },
    },
    h.world.thread.messages[1],
  ];
  h.world.image = false;
  h.render();
  assert.equal(h.world.thread.messages.at(-1), tip);
  assert.equal(h.world.thread.messages.length, length);
  assert.equal(h.state.contextUsage?.estimated, true);
  assert.deepEqual(h.world.recounts, [{ threadId: "thread-a" }]);
  h.render();
  assert.equal(h.world.recounts.length, 1);
});

test("hydrating the saved reasoning settings keeps legacy usage exact", () => {
  const h = harness();
  h.state.reasoningKey = contextUsageReasoningKey({ enable_thinking: true });
  h.state.contextUsage = usage(84);
  h.world.thread.messages = [reply(h.state.contextUsage)];
  h.state.threadScopedSettingsPending = true;
  h.render();
  h.state.reasoningKey = contextUsageReasoningKey({ enable_thinking: false });
  h.state.threadScopedSettingsPending = false;
  h.render();
  assert.equal(h.state.contextUsage?.totalTokens, 84);
  assert.equal(h.state.contextUsage?.estimated, undefined);
  assert.deepEqual(h.world.recounts, []);
});

test("settings pairing preserves an exact count already priced for the restored reasoning mode", () => {
  const h = harness();
  const enabled = contextUsageReasoningKey({ enable_thinking: true });
  h.state.reasoningKey = contextUsageReasoningKey({ enable_thinking: false });
  h.state.threadScopedSettingsPending = true;
  h.world.thread.messages = [reply(usage(84, enabled))];
  h.render();
  const counted = usage(101, enabled);
  h.state.contextUsage = counted;
  h.state.reasoningKey = enabled;
  h.state.threadScopedSettingsPending = false;
  h.render();
  assert.equal(h.state.contextUsage, counted);
  assert.deepEqual(h.writes, []);
  assert.deepEqual(h.world.recounts, []);
});

test("settings pairing cannot replace restored usage with the still-loading empty branch", () => {
  const h = harness();
  h.state.reasoningKey = contextUsageReasoningKey({ enable_thinking: true });
  h.state.threadScopedSettingsPending = true;
  h.world.thread = { isLoading: true, messages: [] };
  h.render();
  const restored = usage(84);
  h.state.contextUsage = restored;
  h.state.threadScopedSettingsPending = false;
  h.state.reasoningKey = contextUsageReasoningKey({ enable_thinking: false });
  h.render();
  assert.equal(h.state.contextUsage, restored);
  assert.deepEqual(h.writes, []);
  assert.deepEqual(h.world.recounts, []);
});

test("finishing history loading restores exact usage after reasoning settings settle", () => {
  const h = harness();
  h.state.reasoningKey = contextUsageReasoningKey({ enable_thinking: false });
  h.state.threadScopedSettingsPending = true;
  h.state.contextUsage = { ...usage(10), estimated: true };
  h.world.thread = { isLoading: true, messages: [] };
  h.render();
  h.world.recounts.length = 0;
  h.state.threadScopedSettingsPending = false;
  h.state.reasoningKey = contextUsageReasoningKey({ enable_thinking: true });
  h.render();
  assert.equal(h.state.contextUsage?.estimated, true);
  h.world.thread = {
    isLoading: false,
    messages: [reply(usage(84, h.state.reasoningKey))],
  };
  h.render();
  assert.equal(h.state.contextUsage?.totalTokens, 84);
  assert.equal(h.state.contextUsage?.estimated, undefined);
  assert.deepEqual(h.world.recounts, []);
});

test("reasoning repricing ignores an outgoing runtime during a thread switch", () => {
  const h = harness();
  h.state.reasoningKey = contextUsageReasoningKey({ enable_thinking: true });
  h.state.contextUsage = usage(42, h.state.reasoningKey);
  h.render();
  const shown = h.state.contextUsage;
  h.world.mainThreadId = "outgoing-thread";
  h.state.reasoningKey = contextUsageReasoningKey({ enable_thinking: false });
  h.render();
  assert.equal(h.state.contextUsage, shown);
  assert.deepEqual(h.writes, []);
  assert.deepEqual(h.world.recounts, []);
});

test("the registered branch belongs to the active thread and must finish loading", () => {
  const h = harness("ActiveBranchRegistrar");
  h.render();
  assert.equal(h.readBranch(), h.world.thread.messages);
  h.world.mainThreadId = "blank-thread";
  h.world.thread.messages = [];
  assert.equal(h.readBranch(), null);
  h.world.mainThreadId = h.state.activeThreadId!;
  assert.equal(h.readBranch(), h.world.thread.messages);
  h.world.thread.isLoading = true;
  assert.equal(h.readBranch(), null);
});
