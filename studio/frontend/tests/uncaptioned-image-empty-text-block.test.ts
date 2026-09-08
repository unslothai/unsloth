// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import ts from "typescript";

// chat-adapter.ts drags in the stores, so lift the source; assistant is a stub.
const adapterSource = readFileSync(
  fileURLToPath(
    new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
  ),
  "utf8",
);

function liftAdapterFunction(opener: string): string {
  const start = adapterSource.indexOf(opener);
  assert.ok(start >= 0, `${opener} is no longer defined in chat-adapter.ts`);
  const end = adapterSource.indexOf("\n}", start);
  assert.ok(end > start, `could not find the end of ${opener}`);
  return adapterSource.slice(start, end + 2);
}

const serializeJs = ts.transpileModule(
  [
    "function serializeAssistantReplayMessages() { throw new Error('not under test'); }",
    liftAdapterFunction("function collectTextParts("),
    liftAdapterFunction("function collectImageParts("),
    liftAdapterFunction("function buildReplayContent("),
    liftAdapterFunction("function toOpenAIMessages("),
    "return { toOpenAIMessages, buildReplayContent };",
  ].join("\n\n"),
  { compilerOptions: { target: ts.ScriptTarget.ES2022 } },
).outputText;

const { toOpenAIMessages, buildReplayContent } = new Function(
  serializeJs,
)() as {
  toOpenAIMessages: (
    message: unknown,
  ) => Array<{ role: string; content: unknown }>;
  buildReplayContent: (text: string, images: unknown[]) => unknown;
};

const IMAGE_DATA_URL = "data:image/png;base64,aGVsbG8=";

test("an uncaptioned image sends no empty text block", () => {
  const [serialized] = toOpenAIMessages({
    role: "user",
    content: [{ type: "image", image: IMAGE_DATA_URL }],
  });

  assert.equal(serialized.role, "user");
  assert.deepEqual(serialized.content, [
    { type: "image_url", image_url: { url: IMAGE_DATA_URL } },
  ]);
});

test("a captioned image still leads with its text block", () => {
  const [serialized] = toOpenAIMessages({
    role: "user",
    content: [
      { type: "text", text: "what is this?" },
      { type: "image", image: IMAGE_DATA_URL },
    ],
  });

  assert.deepEqual(serialized.content, [
    { type: "text", text: "what is this?" },
    { type: "image_url", image_url: { url: IMAGE_DATA_URL } },
  ]);
});

test("a whitespace-only caption sends no text block either", () => {
  // Two empty text parts join to "\n", which is truthy but still rejected.
  const [serialized] = toOpenAIMessages({
    role: "user",
    content: [
      { type: "text", text: "" },
      { type: "text", text: "" },
      { type: "image", image: IMAGE_DATA_URL },
    ],
  });

  assert.deepEqual(serialized.content, [
    { type: "image_url", image_url: { url: IMAGE_DATA_URL } },
  ]);
});

test("a real caption keeps its own surrounding whitespace", () => {
  const [serialized] = toOpenAIMessages({
    role: "user",
    content: [
      { type: "text", text: "  what is this?  " },
      { type: "image", image: IMAGE_DATA_URL },
    ],
  });

  assert.deepEqual(serialized.content, [
    { type: "text", text: "  what is this?  " },
    { type: "image_url", image_url: { url: IMAGE_DATA_URL } },
  ]);
});

test("the serialised content is not the collected image array itself", () => {
  const images = [
    { type: "image_url" as const, image_url: { url: IMAGE_DATA_URL } },
  ];
  assert.notEqual(buildReplayContent("", images), images);
  assert.deepEqual(buildReplayContent("", images), images);
});

test("a text-only turn still serialises to a plain string", () => {
  const [serialized] = toOpenAIMessages({
    role: "user",
    content: [{ type: "text", text: "no attachments here" }],
  });

  assert.equal(serialized.content, "no attachments here");
});
