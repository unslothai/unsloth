// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Deep Research total-time field takes minutes and had no ceiling, while
// ChatSettingsPayload and the run route both cap the seconds it turns into at
// one year. A larger value used to sail through the composer, get dropped from
// the settings patch, and then 400 every run start. These pin the ceiling on
// the paths that write and read the value.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import type { PersistedChatSettings } from "../src/features/chat/api/chat-settings-api.ts";
import {
  assignSanitizedMirroredSettings,
  MAX_RESEARCH_MODEL_TIMEOUT_SECONDS,
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

  // The whole patch is rejected on one out-of-contract field, so what the store
  // holds has to survive sanitisation rather than be dropped from it.
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
