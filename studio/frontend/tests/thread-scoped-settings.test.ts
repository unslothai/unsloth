// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// PATCH /api/chat/threads/{id} is extra="forbid" with range limits; pin what may be sent.

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
  // It disables the sandbox, so it is re-accepted through the warning dialog each session.
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
      // Canvas mode is gone; a chat saved with it on restores without it.
      artifactsEnabled: true,
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
  // loadPermissionMode falls back to it, so a per-chat write would go global.
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
  assert.equal(hasThreadScopedSettings(null), false);
  assert.equal(hasThreadScopedSettings(undefined), false);
  assert.equal(hasThreadScopedSettings({}), false);
  assert.equal(hasThreadScopedSettings({ toolsEnabled: false }), true);
});

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

// extra="forbid" refuses the whole body on one bad field.
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
  assert.deepEqual(
    sanitizeThreadScopedSettings({ temperature: 2, topP: 0, topK: 100 }),
    { temperature: 2, topP: 0, topK: 100 },
  );
});

// -1 disables top-k and many defaults resolve to it, so it must be kept.
test("the disabled top-k value is kept, and it is the floor", () => {
  assert.deepEqual(sanitizeThreadScopedSettings({ topK: -1 }), { topK: -1 });
  assert.deepEqual(sanitizeThreadScopedSettings({ topK: -2 }), {});
});

test("a non-string prompt is dropped rather than coerced", () => {
  assert.deepEqual(
    sanitizeThreadScopedSettings({ systemPrompt: 12, systemVariables: null }),
    {},
  );
  assert.deepEqual(sanitizeThreadScopedSettings({ systemPrompt: "" }), {
    systemPrompt: "",
  });
});

// Context belongs to the loaded model, not the conversation.
test("the context and the model are not per-chat", () => {
  for (const key of ["maxSeqLength", "maxTokens", "checkpoint"]) {
    assert.equal(isThreadScopedSettingKey(key), false, key);
  }
  assert.deepEqual(
    sanitizeThreadScopedSettings({ maxTokens: 4096, checkpoint: "some/model" }),
    {},
  );
});
