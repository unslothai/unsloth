// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const ADAPTER = readSrc("features/chat/api/chat-adapter.ts");
const RUNTIME = readSrc("features/chat/hooks/use-chat-model-runtime.ts");

const FAILURE = ADAPTER.slice(
  ADAPTER.indexOf('toast.error("Generation failed"'),
  ADAPTER.indexOf("// An explicit Stop is an abort too"),
);
const SYNC_FAILURE = RUNTIME.slice(
  RUNTIME.indexOf("export async function syncInferenceStatusToStore("),
  RUNTIME.indexOf('toast.error("Failed to refresh models"'),
);

test("a failed local reply quietly re-reads what the server still holds", () => {
  assert.match(
    FAILURE,
    /if \(!isExternalRequest\) \{\s*void syncInferenceStatusToStore\(\{[^}]*quiet: true/,
  );
  // An idle-unload armed refresh would read the crash reap's empty status as idle-unloaded and keep the dead pick.
  assert.doesNotMatch(FAILURE, /preserveIdleUnloaded/);
  const catchBlock = SYNC_FAILURE.slice(SYNC_FAILURE.lastIndexOf("} catch (error) {"));
  assert.match(catchBlock, /if \(options\?\.quiet\) return;[\s\S]*setModelsError\(message\)/);
});
