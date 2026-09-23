// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const adapter = readSrc("features/chat/api/chat-adapter.ts");
const BODY_END = /\n\s*\};/;
const USAGE_OPT_IN = /stream_options: \{ include_usage: true \},/;

function requestBody(start: RegExp): string {
  const at = adapter.search(start);
  assert.ok(at > 0, `request body starting at ${start} moved`);
  const end = adapter.slice(at).search(BODY_END);
  assert.ok(end > 0, `request body starting at ${start} never closes`);
  return adapter.slice(at, at + end);
}

for (const [name, start] of [
  ["local", /return \{\s*model: params\.checkpoint,/],
  ["connected-provider", /return \{\s*model: externalSelection\.modelId,/],
] as const) {
  test(`the ${name} chat request opts into the stream usage chunk`, () => {
    assert.match(requestBody(start), USAGE_OPT_IN);
  });
}
