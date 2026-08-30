// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import test from "node:test";

import {
  mergedToolCallArgumentsText,
  toolCallArgumentsText,
} from "../src/features/chat/tool-call-arguments.ts";

// The event as it reaches the adapter: JSON.parse has already rounded the id, which is why
// the card cannot be built by re-encoding `arguments`.
const WIRE = '{"arguments":{"id":9007199254740993},"arguments_text":"{\\"id\\":9007199254740993}"}';

test("the card shows the id the tool is run with, not the one JSON.parse returned", () => {
  const event = JSON.parse(WIRE) as {
    arguments: unknown;
    arguments_text: string;
  };
  assert.equal(JSON.stringify(event.arguments), '{"id":9007199254740992}');

  assert.equal(
    toolCallArgumentsText(event.arguments_text, event.arguments),
    '{"id":9007199254740993}',
  );
});

test("an event without the backend's text still renders", () => {
  assert.equal(toolCallArgumentsText(undefined, { q: "gpu" }), '{"q":"gpu"}');
  assert.equal(toolCallArgumentsText("", { q: "gpu" }), '{"q":"gpu"}');
  assert.equal(toolCallArgumentsText(undefined, undefined), "{}");
});

test("completing a call keeps the exact text when nothing merged in", () => {
  const event = JSON.parse(WIRE) as { arguments: unknown; arguments_text: string };
  const exact = toolCallArgumentsText(event.arguments_text, event.arguments);

  // tool_end for a locally executed call carries no further arguments.
  assert.equal(
    mergedToolCallArgumentsText(exact, event.arguments, false),
    '{"id":9007199254740993}',
  );
});

test("a merge that did add arguments is re-encoded, so the text describes them", () => {
  const merged = mergedToolCallArgumentsText(
    '{"id":9007199254740993}',
    { id: 9007199254740993, google: { native_part: 1 } },
    true,
  );

  assert.deepEqual(JSON.parse(merged), {
    id: 9007199254740992,
    google: { native_part: 1 },
  });
});

test("a merge is judged by whether it happened, not by comparing lossy encodings", () => {
  // tool_end setting the id to the value JSON.parse had already rounded it to: the two
  // encodings collide, so anything comparing them would keep text describing the old value.
  const collided = mergedToolCallArgumentsText(
    '{"id":9007199254740993}',
    { id: 9007199254740992 },
    true,
  );

  assert.equal(collided, '{"id":9007199254740992}');
});

// The helpers only hold the line if the adapter actually routes through them: re-encoding
// the parsed arguments at either the tool_start or the tool_end branch would put the
// rounded id back on the card.
const ADAPTER = readFileSync(
  fileURLToPath(new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url)),
  "utf8",
);

test("the adapter builds tool-call argument text through the helpers", () => {
  assert.match(ADAPTER, /toolCallArgumentsText\(\s*toolEvent\.arguments_text/);
  assert.match(ADAPTER, /argsText: mergedToolCallArgumentsText\(/);
  assert.equal(ADAPTER.includes("argsText: JSON.stringify(toolArgs)"), false);
  assert.equal(ADAPTER.includes("argsText: JSON.stringify(mergedArgs ?? {})"), false);
});
