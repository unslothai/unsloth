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

function runtimeFunction(name: string, dependencies: Record<string, unknown>) {
  const declaration = source.statements.find(
    (statement): statement is ts.FunctionDeclaration =>
      ts.isFunctionDeclaration(statement) && statement.name?.text === name,
  );
  assert.ok(declaration, `${name} exists`);
  const { outputText } = ts.transpileModule(declaration.getText(source), {
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
    params: { checkpoint: "local-model" },
    loadedContextLength: 4096,
    loadedIsGguf: true,
    modelLoading: false,
    threadScopedSettingsPending: false,
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
  const component = runtimeFunction(name, {
    ...savedUsage,
    isExternalModelId,
    currentContextUsageReasoningKey: (snapshot = state) => snapshot.reasoningKey,
    useChatRuntimeStore,
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
    messagesContainImage: () => false,
    findLatestUserAudioBase64: () => null,
    findLatestUserVideoBase64: () => null,
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
  assert.equal(h.world.recounts.length, 1);
});

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
