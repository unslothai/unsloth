// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Settings and the run route cap the timeout at one year; over-cap values 400d every run.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import type { PersistedChatSettings } from "../src/features/chat/api/chat-settings-api.ts";
import {
  assignSanitizedMirroredSettings,
  MAX_RESEARCH_MODEL_TIMEOUT_SECONDS,
  MIN_FINITE_RESEARCH_MODEL_TIMEOUT_SECONDS,
  sanitizeBoundedNumber,
} from "../src/features/chat/utils/mirrored-chat-settings.ts";
import { installLocalStorageFake } from "./helpers/kit.ts";

const { store: localStorageFake } = installLocalStorageFake();
localStorageFake.set("unsloth_chat_settings_imported_to_studio_db", "true");
register("./store-settings-resolver.mjs", import.meta.url);

const { useChatRuntimeStore, DEFAULT_RESEARCH_MODEL_TIMEOUT_SECONDS } =
  await import("../src/features/chat/stores/chat-runtime-store.ts");

test("the cap matches the ceiling the backend payload enforces", () => {
  assert.equal(MAX_RESEARCH_MODEL_TIMEOUT_SECONDS, 365 * 24 * 3600);
});

test("the mirrored patch keeps the cap and drops one second past it", () => {
  const atCap: PersistedChatSettings = {};
  assignSanitizedMirroredSettings(
    { researchModelTimeoutSeconds: MAX_RESEARCH_MODEL_TIMEOUT_SECONDS },
    atCap,
  );
  assert.equal(
    atCap.researchModelTimeoutSeconds,
    MAX_RESEARCH_MODEL_TIMEOUT_SECONDS,
  );

  const overCap: PersistedChatSettings = {};
  assignSanitizedMirroredSettings(
    { researchModelTimeoutSeconds: MAX_RESEARCH_MODEL_TIMEOUT_SECONDS + 1 },
    overCap,
  );
  assert.equal(overCap.researchModelTimeoutSeconds, undefined);
});

test("an over-cap budget never reaches the store or storage", () => {
  const store = useChatRuntimeStore.getState();

  store.setResearchModelTimeoutSeconds(1000000 * 60);
  assert.equal(
    useChatRuntimeStore.getState().researchModelTimeoutSeconds,
    DEFAULT_RESEARCH_MODEL_TIMEOUT_SECONDS,
  );

  // One out-of-contract field rejects the whole patch.
  const mirrored: PersistedChatSettings = {};
  assignSanitizedMirroredSettings(
    {
      researchModelTimeoutSeconds:
        useChatRuntimeStore.getState().researchModelTimeoutSeconds,
    },
    mirrored,
  );
  assert.equal(
    mirrored.researchModelTimeoutSeconds,
    DEFAULT_RESEARCH_MODEL_TIMEOUT_SECONDS,
  );

  store.setResearchModelTimeoutSeconds(MAX_RESEARCH_MODEL_TIMEOUT_SECONDS);
  assert.equal(
    useChatRuntimeStore.getState().researchModelTimeoutSeconds,
    MAX_RESEARCH_MODEL_TIMEOUT_SECONDS,
  );
  store.setResearchModelTimeoutSeconds(0);
  assert.equal(useChatRuntimeStore.getState().researchModelTimeoutSeconds, 0);
});

// 0 is the unlimited sentinel; finite values below the route's floor of 10 would 400.
test("a sub-floor finite timeout is refused on every frontend path", () => {
  const store = useChatRuntimeStore.getState();

  for (const rejected of [1, 5, 9]) {
    const mirrored: PersistedChatSettings = {};
    assignSanitizedMirroredSettings(
      { researchModelTimeoutSeconds: rejected },
      mirrored,
    );
    assert.equal(mirrored.researchModelTimeoutSeconds, undefined);

    store.setResearchModelTimeoutSeconds(rejected);
    assert.equal(
      useChatRuntimeStore.getState().researchModelTimeoutSeconds,
      DEFAULT_RESEARCH_MODEL_TIMEOUT_SECONDS,
    );
  }

  for (const accepted of [0, MIN_FINITE_RESEARCH_MODEL_TIMEOUT_SECONDS]) {
    const mirrored: PersistedChatSettings = {};
    assignSanitizedMirroredSettings(
      { researchModelTimeoutSeconds: accepted },
      mirrored,
    );
    assert.equal(mirrored.researchModelTimeoutSeconds, accepted);
  }

  const bounds = { min: 0, minPositive: 10, max: 100, integer: true };
  assert.equal(sanitizeBoundedNumber(0, bounds), 0);
  assert.equal(sanitizeBoundedNumber(5, bounds), undefined);
  assert.equal(sanitizeBoundedNumber(10, bounds), 10);
});

// The max attribute does not stop typed values, so clamp rather than fall back to default.
test("an over-cap typed value saves as the cap, not as the default", () => {
  const store = useChatRuntimeStore.getState();
  const maxMinutes = Math.floor(MAX_RESEARCH_MODEL_TIMEOUT_SECONDS / 60);

  // What the dialog's save handler computes for a typed 1000000 minutes.
  const saved = Math.min(1000000, maxMinutes) * 60;
  assert.equal(saved, MAX_RESEARCH_MODEL_TIMEOUT_SECONDS);

  store.setResearchModelTimeoutSeconds(saved);
  assert.equal(
    useChatRuntimeStore.getState().researchModelTimeoutSeconds,
    MAX_RESEARCH_MODEL_TIMEOUT_SECONDS,
  );
  assert.notEqual(
    useChatRuntimeStore.getState().researchModelTimeoutSeconds,
    DEFAULT_RESEARCH_MODEL_TIMEOUT_SECONDS,
  );
});
