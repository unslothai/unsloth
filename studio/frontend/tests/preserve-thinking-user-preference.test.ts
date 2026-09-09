// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// preserveThinking is the installation's own switch, persisted through
// PUT /api/chat/settings, and the backend now also advertises a per-family default
// (on for Qwen3.8, off elsewhere). The default has to seed the switch without ever
// replacing an answer the user gave: otherwise a cold boot is a coin flip between the
// settings GET and the inference status, and switching model families silently
// re-enables a toggle that was turned off. These drive the real store, whose recorded
// answer is module state, so they run in order and only ever add to it.

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
  // A template without the kwarg, and a backend too old to advertise the field.
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
  // Startup with a resident Qwen3.8: the status lands first, so the store has only
  // the family default to seed from.
  useChatRuntimeStore.setState({
    preserveThinking: resolvePreserveThinkingOnLoad(QWEN38),
  });
  assert.equal(useChatRuntimeStore.getState().preserveThinking, true);
  settingsHttp.release?.();
  await hydrating;
  // The seed advanced no mutation version, so hydration still applies the answer.
  assert.equal(useChatRuntimeStore.getState().preserveThinking, false);
});

test("and the other order lands on the same value, not the opposite one", () => {
  // The regression: the status applier publishes the family default on every
  // model/variant change, so with hydration already done it replaced the stored
  // false with true and every request from then on carried the prior reasoning.
  assert.equal(resolvePreserveThinkingOnLoad(QWEN38), false);
});

test("returning to Qwen3.8 from another family keeps the stored answer", () => {
  // Away to a Qwen3.6 and back. reloadingSameModel is false on both loads, so
  // performLoad resolves the value rather than carrying the pre-unload one over.
  assert.equal(resolvePreserveThinkingOnLoad(QWEN36), false);
  assert.equal(resolvePreserveThinkingOnLoad(QWEN38), false);
});

test("the composer toggle answers for the installation too", () => {
  useChatRuntimeStore.getState().setPreserveThinking(true);
  // Including against a family whose default is off, which would otherwise turn a
  // toggle the user just switched on back off at the next model switch.
  assert.equal(resolvePreserveThinkingOnLoad(QWEN36), true);
  assert.equal(resolvePreserveThinkingOnLoad(QWEN38), true);
});

// The writers below cannot be driven from here (a .tsx barrel sits in their import
// graph), so their wiring is pinned against the source, as the sibling store suites do.
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
