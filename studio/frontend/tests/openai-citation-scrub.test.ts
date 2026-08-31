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

test("orphan private-use bytes are still removed", () => {
  assert.equal(scrubOpenAICitationMarkers("a\uE201b"), "ab");
  assert.equal(scrubOpenAICitationMarkers("a\uE202b"), "ab");
});

test("text without markers is returned untouched", () => {
  assert.equal(scrubOpenAICitationMarkers("plain text"), "plain text");
  assert.equal(scrubOpenAICitationMarkers(""), "");
});

test("a lone private-use glyph costs one character, not the rest of the message", () => {
  // Nerd Font icons live in the same private-use block, so a partial rule that keys on
  // the open byte alone would swallow everything after a prompt-icon example.
  assert.equal(
    scrubOpenAICitationMarkers("icon \uE200 and the rest"),
    "icon  and the rest",
  );
  assert.equal(scrubOpenAICitationMarkers("trailing \uE200"), "trailing ");
  assert.equal(
    scrubOpenAICitationMarkers("\uE200not a marker"),
    "not a marker",
  );
});

test("every truncation of the marker prefix is still dropped", () => {
  for (const partial of [
    "\uE200c",
    "\uE200ci",
    "\uE200cit",
    "\uE200cite",
    "\uE200cite\uE202",
    PARTIAL,
  ]) {
    assert.equal(scrubOpenAICitationMarkers(`See ${partial}`), "See ");
  }
});
