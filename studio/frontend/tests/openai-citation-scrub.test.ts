// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { scrubOpenAICitationMarkers } from "../src/components/assistant-ui/openai-citation-scrub.ts";

const MARKER = "\uE200cite\uE202turn0search0\uE201";
const PARTIAL = "\uE200cite\uE202turn0search0";

test("a complete marker leaves no trace", () => {
  assert.equal(
    scrubOpenAICitationMarkers(`See ${MARKER} for more`),
    "See  for more",
  );
});

test("a truncated marker takes its payload with it", () => {
  assert.equal(scrubOpenAICitationMarkers(`See ${PARTIAL}`), "See ");
  assert.equal(scrubOpenAICitationMarkers(`a ${MARKER} b ${PARTIAL}`), "a  b ");
});

test("a truncation earlier than the payload is dropped too", () => {
  assert.equal(scrubOpenAICitationMarkers("See \uE200ci"), "See ");
  assert.equal(scrubOpenAICitationMarkers("See \uE200"), "See ");
});

test("orphan private-use bytes are still removed", () => {
  assert.equal(scrubOpenAICitationMarkers("a\uE201b"), "ab");
  assert.equal(scrubOpenAICitationMarkers("a\uE202b"), "ab");
});

test("text without markers is returned untouched", () => {
  assert.equal(scrubOpenAICitationMarkers("plain text"), "plain text");
  assert.equal(scrubOpenAICitationMarkers(""), "");
});
