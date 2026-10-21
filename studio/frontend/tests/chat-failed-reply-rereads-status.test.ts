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

test("a failed local reply re-reads what the server still holds", () => {
  assert.match(RUNTIME, /export async function syncInferenceStatusToStore\(/);
  assert.match(
    FAILURE,
    /if \(!isExternalRequest\) \{\s*void syncInferenceStatusToStore\(/,
  );
});
