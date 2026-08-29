// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The snapshot goes out on PATCH /api/chat/threads/{id}, whose model is extra="forbid" with
// literal and range constraints. These pin what the client may send, and what may be per-chat.

import assert from "node:assert/strict";
import test from "node:test";

import {
  THREAD_SCOPED_SETTING_KEYS,
  hasThreadScopedSettings,
  isThreadOwnedSettingKey,
  isThreadScopedSettingKey,
  sanitizeThreadScopedSettings,
} from "../src/features/chat/utils/thread-scoped-settings.ts";

test("a full snapshot survives the round trip", () => {
  const settings = sanitizeThreadScopedSettings({
    reasoningEnabled: true,
    reasoningEffort: "high",
    toolsEnabled: true,
    codeToolsEnabled: false,
    imageToolsEnabled: false,
    webFetchToolsEnabled: true,
    deepResearchEnabled: false,
    artifactsEnabled: true,
    mcpEnabledForChat: false,
    permissionMode: "auto",
    ragEnabled: true,
    ragSource: { type: "kb", kbId: "notes" },
    ragMode: "dense",
    ragTopK: 12,
    ragAutoInject: "on",
    ragAutoInjectMinScore: 0.42,
  });

  assert.deepEqual(settings, {
    reasoningEnabled: true,
    reasoningEffort: "high",
    toolsEnabled: true,
    codeToolsEnabled: false,
    imageToolsEnabled: false,
    webFetchToolsEnabled: true,
    deepResearchEnabled: false,
    artifactsEnabled: true,
    mcpEnabledForChat: false,
    permissionMode: "auto",
    ragEnabled: true,
    ragSource: { type: "kb", kbId: "notes" },
    ragMode: "dense",
    ragTopK: 12,
    ragAutoInject: "on",
    ragAutoInjectMinScore: 0.42,
  });
});

test("full access is dropped rather than stored on the thread", () => {
  // it disables the sandbox, so it is re-accepted through the warning dialog each session.
  assert.deepEqual(
    sanitizeThreadScopedSettings({ permissionMode: "full" }),
    {},
  );
});

test("out-of-contract values are dropped", () => {
  assert.deepEqual(
    sanitizeThreadScopedSettings({
      toolsEnabled: "yes",
      ragMode: "vector",
      ragTopK: 51,
      ragAutoInjectMinScore: 1.5,
      ragSource: { type: "kb" },
      reasoningEffort: "extreme",
    }),
    {},
  );
});

test("settings that describe the installation stay out of the snapshot", () => {
  // these belong to the install, so a chat must not start pinning its own copy of them.
  for (const key of [
    "showCanvasMenuItem",
    "collapseHtmlArtifacts",
    "allowArtifactNetworkAccess",
    "searchImages",
    "ragOcrScanned",
    "ragCaptionFigures",
    "researchWebsitePolicy",
    "researchModelTimeoutSeconds",
    "speculativeType",
    "gpuMemoryMode",
    "expandQuantizations",
    "showAllQuantizations",
    "fitOnDeviceOnly",
    "autoTitle",
  ]) {
    assert.equal(isThreadScopedSettingKey(key), false, key);
  }
  assert.deepEqual(
    sanitizeThreadScopedSettings({
      gpuMemoryMode: "manual",
      showCanvasMenuItem: true,
      ragOcrScanned: true,
    }),
    {},
  );
});

test("every thread-scoped key is recognised and non-object input is safe", () => {
  for (const key of THREAD_SCOPED_SETTING_KEYS) {
    assert.equal(isThreadScopedSettingKey(key), true, key);
  }
  assert.deepEqual(sanitizeThreadScopedSettings(null), {});
  assert.deepEqual(sanitizeThreadScopedSettings("toolsEnabled"), {});
  assert.deepEqual(sanitizeThreadScopedSettings([1, 2]), {});
});

test("the legacy confirm toggle is owned by the chat but not stored on it", () => {
  // loadPermissionMode falls back to it, so a per-chat change that wrote it would go global.
  assert.equal(isThreadOwnedSettingKey("confirmToolCalls"), true);
  assert.equal(isThreadScopedSettingKey("confirmToolCalls"), false);
  assert.deepEqual(
    sanitizeThreadScopedSettings({ confirmToolCalls: true }),
    {},
  );
  for (const key of THREAD_SCOPED_SETTING_KEYS) {
    assert.equal(isThreadOwnedSettingKey(key), true, key);
  }
  assert.equal(isThreadOwnedSettingKey("gpuMemoryMode"), false);
});

test("an empty snapshot reads as no snapshot", () => {
  // a thread that stored nothing falls back to the installation settings, as chats did before.
  assert.equal(hasThreadScopedSettings(null), false);
  assert.equal(hasThreadScopedSettings(undefined), false);
  assert.equal(hasThreadScopedSettings({}), false);
  assert.equal(hasThreadScopedSettings({ toolsEnabled: false }), true);
});

// The reported gap: returning to a chat started under one system prompt showed whichever
// prompt the last chat had. These live under `params`, which is all that makes them special.
test("the sampling params and the system prompt travel with the chat", () => {
  const settings = sanitizeThreadScopedSettings({
    temperature: 0.2,
    topP: 0.85,
    topK: 40,
    minP: 0.02,
    repetitionPenalty: 1.1,
    presencePenalty: 0.5,
    systemPrompt: "You are a terse reviewer.",
    systemVariables: "name=Ada",
  });
  assert.deepEqual(settings, {
    temperature: 0.2,
    topP: 0.85,
    topK: 40,
    minP: 0.02,
    repetitionPenalty: 1.1,
    presencePenalty: 0.5,
    systemPrompt: "You are a terse reviewer.",
    systemVariables: "name=Ada",
  });
  for (const key of Object.keys(settings)) {
    assert.ok(isThreadScopedSettingKey(key), key);
    assert.ok(isThreadOwnedSettingKey(key), key);
  }
});

// The bounds are the PATCH model's, so a value the server would refuse must not
// be sent: extra="forbid" refuses the whole body on one bad field.
test("a sampling value outside the slider range is dropped", () => {
  assert.deepEqual(
    sanitizeThreadScopedSettings({
      temperature: 2.5,
      topP: -0.1,
      topK: 101,
      minP: 2,
      repetitionPenalty: 0.5,
      presencePenalty: 3,
    }),
    {},
  );
  // The edges themselves are inside.
  assert.deepEqual(
    sanitizeThreadScopedSettings({ temperature: 2, topP: 0, topK: 100 }),
    { temperature: 2, topP: 0, topK: 100 },
  );
});

// -1 disables top-k, and default.yaml and whole model families resolve to it, so dropping
// it means reopening such a chat silently takes whatever top-k the installation last saw.
test("the disabled top-k value is kept, and it is the floor", () => {
  assert.deepEqual(sanitizeThreadScopedSettings({ topK: -1 }), { topK: -1 });
  assert.deepEqual(sanitizeThreadScopedSettings({ topK: -2 }), {});
});

test("a non-string prompt is dropped rather than coerced", () => {
  assert.deepEqual(
    sanitizeThreadScopedSettings({ systemPrompt: 12, systemVariables: null }),
    {},
  );
  // An empty prompt is a real choice, not a missing one.
  assert.deepEqual(sanitizeThreadScopedSettings({ systemPrompt: "" }), {
    systemPrompt: "",
  });
});

// Context belongs to the model that loaded, not to the conversation, so a chat
// restoring a budget the current model cannot hold is not a thing that happens.
test("the context and the model are not per-chat", () => {
  for (const key of ["maxSeqLength", "maxTokens", "checkpoint"]) {
    assert.equal(isThreadScopedSettingKey(key), false, key);
  }
  assert.deepEqual(
    sanitizeThreadScopedSettings({ maxTokens: 4096, checkpoint: "some/model" }),
    {},
  );
});
