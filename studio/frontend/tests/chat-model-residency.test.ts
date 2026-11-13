// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Loading an image or video model evicts the chat model without changing the selection.

import assert from "node:assert/strict";
import test from "node:test";

import { chatModelLoaded } from "../src/features/chat/lib/chat-model-loaded.ts";

import { readSrc } from "./helpers/kit.ts";

const USE_CHAT_MODEL_RUNTIME = readSrc("features/chat/hooks/use-chat-model-runtime.ts");

const PICKED = "unsloth/Qwen3.5-9B-GGUF";

test("a resident model reads as loaded", () => {
  assert.equal(
    chatModelLoaded({
      checkpoint: PICKED,
      modelLoading: false,
      isExternalModel: false,
      residentCheckpoint: PICKED,
    }),
    true,
  );
});

test("a model evicted for an image load does not read as loaded", () => {
  assert.equal(
    chatModelLoaded({
      checkpoint: PICKED,
      modelLoading: false,
      isExternalModel: false,
      residentCheckpoint: null,
    }),
    false,
  );
});

test("residency not yet read is not treated as evicted", () => {
  assert.equal(
    chatModelLoaded({
      checkpoint: PICKED,
      modelLoading: false,
      isExternalModel: false,
      residentCheckpoint: undefined,
    }),
    true,
  );
});

test("an external model is loaded whatever the backend holds", () => {
  assert.equal(
    chatModelLoaded({
      checkpoint: "openai:gpt-5",
      modelLoading: false,
      isExternalModel: true,
      residentCheckpoint: null,
    }),
    true,
  );
});

test("nothing picked is never loaded", () => {
  assert.equal(
    chatModelLoaded({
      checkpoint: "",
      modelLoading: false,
      isExternalModel: false,
      residentCheckpoint: PICKED,
    }),
    false,
  );
});

test("the selector's tick asks the caller, and defaults to the old rule", () => {
  const source = readSrc("features/model-picker/components/model-selector.tsx");
  assert.match(
    source,
    /const isLoaded = selected !== "" && \(loaded \?\? true\)/,
  );
  const page = readSrc("features/chat/chat-page.tsx");
  assert.match(
    page,
    /loaded=\{chatModelLoaded\(\{/,
    "the chat header must pass it",
  );
  assert.match(page, /residentCheckpoint,/);
});

test("a model still loading is not loaded yet", () => {
  assert.equal(
    chatModelLoaded({
      checkpoint: PICKED,
      modelLoading: true,
      isExternalModel: false,
      residentCheckpoint: PICKED,
    }),
    false,
  );
});

test("the picker's Loaded badge asks residency, not the selection", () => {
  const pickers = readSrc("features/model-picker/components/model-selector/pickers.tsx");
  assert.match(pickers, /const chatLoadedModelId = chatModelLoaded\(\{/);
  assert.match(
    pickers,
    /const loadedModelId = loadedModelIdOverride \?\? chatLoadedModelId/,
  );
  assert.match(pickers, /residentCheckpoint,/);
  assert.doesNotMatch(
    pickers,
    /const loadedModelId = useChatRuntimeStore\(\(s\) => s\.params\.checkpoint\)/,
  );
});

// Nothing polls /status, and the arbiter evicts chat at the start of another runtime's load,
// so the re-read must be driven by that load starting.
test("another runtime loading re-reads the chat status", () => {
  assert.match(USE_CHAT_MODEL_RUNTIME, /subscribeModelLifecycle\(\(\{ runtime \}\) => \{/);
  assert.match(USE_CHAT_MODEL_RUNTIME, /if \(runtime === "chat" \|\| runtime === "stt"\) return;/);
  assert.doesNotMatch(
    USE_CHAT_MODEL_RUNTIME,
    /if \(loading \|\| runtime === "chat"\) return;/,
    "the settle-only guard is what left the picker naming an evicted model",
  );
  assert.match(
    USE_CHAT_MODEL_RUNTIME,
    /void refresh\(\{\s*includeLoras: false,\s*externalChatSlotLoad: runtime === "tts",\s*\}\)/,
  );
  assert.match(USE_CHAT_MODEL_RUNTIME, /residentCheckpoint: null,/);
});

test("an eviction drops the pick, not just the loaded marks", () => {
  // Anchored on the branch and matched loosely, since the guard has been reflowed before.
  const branchStart = USE_CHAT_MODEL_RUNTIME.search(/\} else if \(\s*!chatActiveModel/);
  assert.notEqual(branchStart, -1, "the eviction branch anchor no longer matches");
  const branch = USE_CHAT_MODEL_RUNTIME.slice(
    branchStart,
    USE_CHAT_MODEL_RUNTIME.indexOf("} catch (error) {", branchStart),
  );
  assert.match(branch, /clearCheckpoint\(\)/);
  assert.match(
    branch,
    /\(wasResident \|\| isSpeechOnlyStatus\(statusRes\)\)[\s\S]*selectedCheckpoint[\s\S]*!modelLoading/,
  );
});

test("the eviction clear reads the store's loading flag", () => {
  const store = readSrc("features/chat/stores/chat-runtime-store.ts");
  assert.match(store, /modelLoading: boolean;/);
  assert.match(store, /set\(\{ modelLoading: true \}\)/);
});
