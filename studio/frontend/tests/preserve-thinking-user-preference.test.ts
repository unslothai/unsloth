// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The per-family default seeds the switch but never replaces a user answer. Recorded answer is
// module state, so tests run in order.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

const { store: localStorageFake } = installLocalStorageFake();
localStorageFake.set("unsloth_chat_settings_imported_to_studio_db", "true");
register("./store-settings-resolver.mjs", import.meta.url);

const { settingsHttp } = await import("./helpers/store-stubs/settings-http.ts");
const { resolvePreserveThinkingOnLoad, useChatRuntimeStore } = await import(
  "../src/features/chat/stores/chat-runtime-store.ts"
);

const QWEN38 = {
  supports_preserve_thinking: true,
  preserve_thinking_default: true,
};
const QWEN36 = {
  supports_preserve_thinking: true,
  preserve_thinking_default: false,
};

test("with no answer of its own the installation takes the family default", () => {
  assert.equal(resolvePreserveThinkingOnLoad(QWEN38), true);
  assert.equal(resolvePreserveThinkingOnLoad(QWEN36), false);
  assert.equal(
    resolvePreserveThinkingOnLoad({
      supports_preserve_thinking: false,
      preserve_thinking_default: true,
    }),
    false,
  );
  assert.equal(
    resolvePreserveThinkingOnLoad({ supports_preserve_thinking: true }),
    false,
  );
});

test("a status that overtakes the settings GET does not strand the stored answer", async () => {
  settingsHttp.settings = { preserveThinking: false };
  settingsHttp.hold();
  const hydrating = useChatRuntimeStore.getState().hydratePersistedSettings();
  useChatRuntimeStore.setState({
    preserveThinking: resolvePreserveThinkingOnLoad(QWEN38),
  });
  assert.equal(useChatRuntimeStore.getState().preserveThinking, true);
  settingsHttp.release?.();
  await hydrating;
  assert.equal(useChatRuntimeStore.getState().preserveThinking, false);
});

test("and the other order lands on the same value, not the opposite one", () => {
  // The status applier publishes the family default on every model change.
  assert.equal(resolvePreserveThinkingOnLoad(QWEN38), false);
});

test("returning to Qwen3.8 from another family keeps the stored answer", () => {
  assert.equal(resolvePreserveThinkingOnLoad(QWEN36), false);
  assert.equal(resolvePreserveThinkingOnLoad(QWEN38), false);
});

test("the composer toggle answers for the installation too", () => {
  useChatRuntimeStore.getState().setPreserveThinking(true);
  assert.equal(resolvePreserveThinkingOnLoad(QWEN36), true);
  assert.equal(resolvePreserveThinkingOnLoad(QWEN38), true);
});

// A .tsx barrel is in these writers' import graph, so pin their wiring against source.
test("every load and status writer resolves rather than taking the raw default", () => {
  for (const path of [
    "../src/features/chat/lib/apply-inference-status-to-store.ts",
    "../src/features/chat/hooks/use-chat-model-runtime.ts",
    "../src/features/chat/api/chat-adapter.ts",
    "../src/features/chat/shared-composer.tsx",
  ]) {
    const source = readFileSync(new URL(path, import.meta.url), "utf8");
    assert.match(source, /resolvePreserveThinkingOnLoad\(/, path);
    assert.doesNotMatch(source, /preserveThinkingDefaultFromLoad\(/, path);
  }
});
