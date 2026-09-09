// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Deep Research total-time minutes field had no ceiling, while ChatSettingsPayload and
// the run route cap the seconds it turns into at one year: an over-cap value was dropped
// from the settings patch and then 400d every run. These pin the ceiling on every path.

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

  // 1000000 minutes, which the composer's unbounded minutes field accepted.
  store.setResearchModelTimeoutSeconds(1000000 * 60);
  assert.equal(
    useChatRuntimeStore.getState().researchModelTimeoutSeconds,
    DEFAULT_RESEARCH_MODEL_TIMEOUT_SECONDS,
  );

  // One out-of-contract field rejects the whole patch, so what the store holds
  // has to survive sanitisation rather than be dropped from it.
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

  // The cap itself and the unlimited sentinel still round trip.
  store.setResearchModelTimeoutSeconds(MAX_RESEARCH_MODEL_TIMEOUT_SECONDS);
  assert.equal(
    useChatRuntimeStore.getState().researchModelTimeoutSeconds,
    MAX_RESEARCH_MODEL_TIMEOUT_SECONDS,
  );
  store.setResearchModelTimeoutSeconds(0);
  assert.equal(useChatRuntimeStore.getState().researchModelTimeoutSeconds, 0);
});

// 0 is the unlimited sentinel, so it is legal below the run route's finite floor of 10.
// Anything between the two would hydrate, be sent unchanged, and 400 every run.
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

  // The sentinel and the floor itself stay legal.
  for (const accepted of [0, MIN_FINITE_RESEARCH_MODEL_TIMEOUT_SECONDS]) {
    const mirrored: PersistedChatSettings = {};
    assignSanitizedMirroredSettings(
      { researchModelTimeoutSeconds: accepted },
      mirrored,
    );
    assert.equal(mirrored.researchModelTimeoutSeconds, accepted);
  }

  // The shared sanitizer keeps the rule for any caller, not just this field.
  const bounds = { min: 0, minPositive: 10, max: 100, integer: true };
  assert.equal(sanitizeBoundedNumber(0, bounds), 0);
  assert.equal(sanitizeBoundedNumber(5, bounds), undefined);
  assert.equal(sanitizeBoundedNumber(10, bounds), 10);
});

// The max attribute does not stop a typed value reaching the save handler, so it clamps:
// falling through to the default would hand someone asking for a long run a short one.
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
