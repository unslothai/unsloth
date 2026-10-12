// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import ts from "typescript";

import { readSrc } from "./helpers/kit.ts";

const adapterSource = readSrc("features/chat/api/chat-adapter.ts");

function liftAdapterFunction(opener: string): string {
  const start = adapterSource.indexOf(opener);
  assert.ok(start >= 0, `${opener} is no longer defined in chat-adapter.ts`);
  const end = adapterSource.indexOf("\n}", start);
  assert.ok(end > start, `could not find the end of ${opener}`);
  return adapterSource.slice(start, end + 2);
}

const serializeJs = ts.transpileModule(
  [
    "function modelVisibleMessage(message) { return message; }",
    "function serializeAssistantReplayMessages() { throw new Error('not under test'); }",
    liftAdapterFunction("function collectTextParts("),
    liftAdapterFunction("function collectImageParts("),
    liftAdapterFunction("function buildReplayContent("),
    liftAdapterFunction("function toOpenAIMessages("),
    "return { toOpenAIMessages };",
  ].join("\n\n"),
  { compilerOptions: { target: ts.ScriptTarget.ES2022 } },
).outputText;

const { toOpenAIMessages } = new Function(serializeJs)() as {
  toOpenAIMessages: (
    message: unknown,
  ) => Array<{ role: string; content: unknown }>;
};

function replayedImageUrl(image: string): string {
  const [serialized] = toOpenAIMessages({
    role: "user",
    content: [
      { type: "text", text: "what is this?" },
      { type: "image", image },
    ],
  });
  const parts = serialized.content as Array<{
    type: string;
    image_url?: { url: string };
  }>;
  return parts[1].image_url!.url;
}

test("an imported https image is replayed as the same url", () => {
  const url = "https://example.com/photos/cat.jpg";
  assert.equal(replayedImageUrl(url), url);
});

test("an http image url is replayed unchanged too", () => {
  const url = "HTTP://example.com/cat.png";
  assert.equal(replayedImageUrl(url), url);
});

test("a data url image is replayed unchanged", () => {
  const url = "data:image/jpeg;base64,aGVsbG8=";
  assert.equal(replayedImageUrl(url), url);
});

test("bare base64 is still wrapped as a png data url", () => {
  assert.equal(replayedImageUrl("aGVsbG8="), "data:image/png;base64,aGVsbG8=");
});
