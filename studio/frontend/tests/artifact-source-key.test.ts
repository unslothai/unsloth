// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  createArtifactId,
  createChatArtifact,
  hashArtifactCode,
} from "../src/features/chat/artifacts/types.ts";

// The source view keys its Streamdown on `${artifact.id}:${hashArtifactCode(artifact.code)}`.
// Streamdown never revises a block it has already committed, so that key must
// change whenever the rendered code does, or the panel keeps showing the
// previous artifact's source.
const sourceKey = (artifact: { id: string; code: string }): string =>
  `${artifact.id}:${hashArtifactCode(artifact.code)}`;

const toolInput = (code: string) => ({
  code,
  source: "tool" as const,
  threadId: "thread-1",
  sourceMessageId: "msg-1",
  sourceToolCallId: "call_0",
});

const fenceInput = (code: string) => ({
  code,
  source: "fence" as const,
  threadId: "thread-1",
  sourceMessageId: "msg-1",
});

test("tool artifact IDs are stable across code changes, so the ID alone is not enough", () => {
  const first = createArtifactId(toolInput("<p>first</p>"));
  const second = createArtifactId(toolInput("<p>second</p>"));
  assert.equal(first, second);
});

test("the source key changes when a tool artifact's code changes", () => {
  const first = createChatArtifact(toolInput("<p>first</p>"));
  const second = createChatArtifact(toolInput("<p>second</p>"));
  assert.notEqual(sourceKey(first), sourceKey(second));
});

test("the source key changes when switching between fence artifacts", () => {
  const first = createChatArtifact(fenceInput("<p>alpha</p>"));
  const second = createChatArtifact(fenceInput("<p>bravo</p>"));
  assert.notEqual(sourceKey(first), sourceKey(second));
});

test("the source key is stable for an unchanged artifact, so no needless remount", () => {
  const code = "<p>same</p>";
  assert.equal(
    sourceKey(createChatArtifact(toolInput(code))),
    sourceKey(createChatArtifact(toolInput(code))),
  );
});

test("hashArtifactCode separates same-length codes and empty from whitespace", () => {
  assert.notEqual(hashArtifactCode("<p>ab</p>"), hashArtifactCode("<p>ba</p>"));
  assert.notEqual(hashArtifactCode(""), hashArtifactCode(" "));
});
