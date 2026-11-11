// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A send that waits on a model load must not pick up the settings of the chat opened meanwhile.

import assert from "node:assert/strict";
import test from "node:test";

import {
  THREAD_SCOPED_PARAM_KEYS,
  THREAD_SCOPED_SETTING_KEYS,
  isThreadScopedParamKey,
  withRunThreadScopedSettings,
} from "../src/features/chat/utils/thread-scoped-settings.ts";

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
