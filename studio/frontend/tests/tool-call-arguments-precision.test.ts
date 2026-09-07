// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  mergedToolCallArgumentsText,
  toolCallArgumentsText,
  toolCallReplayArguments,
} from "../src/features/chat/tool-call-arguments.ts";

const WIRE =
  '{"arguments":{"id":9007199254740993},"arguments_text":"{\\"id\\":9007199254740993}"}';

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
  const event = JSON.parse(WIRE) as {
    arguments: unknown;
    arguments_text: string;
  };
  const exact = toolCallArgumentsText(event.arguments_text, event.arguments);

  assert.equal(
    mergedToolCallArgumentsText(exact, event.arguments),
    '{"id":9007199254740993}',
  );
});

test("a merge adds metadata without rewriting the executed integer", () => {
  const merged = mergedToolCallArgumentsText('{"id":9007199254740993}', {
    id: 9007199254740992,
    google: { native_part: 1 },
  });

  assert.equal(merged, '{"id":9007199254740993,"google":{"native_part":1}}');
});

test("an explicitly overwritten key uses the tool_end value", () => {
  // The two encodings collide, so comparing them would keep text describing the old value.
  const collided = mergedToolCallArgumentsText(
    '{"id":9007199254740993}',
    { id: 9007199254740992 },
    ["id"],
  );

  assert.equal(collided, '{"id":9007199254740992}');
});

test("nested metadata appends without rewriting earlier numeric lexemes", () => {
  const merged = mergedToolCallArgumentsText(
    '{"google":{"native_part":{"parts":[{"result":{"id":9007199254740993}}]}}}',
    {
      google: {
        native_part: {
          parts: [
            { result: { id: 9007199254740992 } },
            { inlineData: { mimeType: "image/png" } },
          ],
        },
      },
    },
  );
  assert.equal(
    merged,
    '{"google":{"native_part":{"parts":[{"result":{"id":9007199254740993}},{"inlineData":{"mimeType":"image/png"}}]}}}',
  );
});

test("prompt replay prefers exact parsable argument text", () => {
  assert.equal(
    toolCallReplayArguments('{"id":9007199254740993}', {
      id: 9007199254740992,
    }),
    '{"id":9007199254740993}',
  );
});

// Read as source: the helpers only hold the line if the adapter routes through them.
const ADAPTER = readFileSync(
  fileURLToPath(
    new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
  ),
  "utf8",
);
const PROMPT_STORAGE = readFileSync(
  fileURLToPath(
    new URL(
      "../src/features/chat/prompt-storage/prompt-storage-dialog.tsx",
      import.meta.url,
    ),
  ),
  "utf8",
);

test("the adapter builds tool-call argument text through the helpers", () => {
  assert.match(
    ADAPTER,
    /toolCallArgumentsText\(\s*hadReservedMetadata \? undefined : toolEvent\.arguments_text/,
  );
  assert.match(ADAPTER, /argsText: mergedToolCallArgumentsText\(/);
  assert.equal(ADAPTER.includes("argsText: JSON.stringify(toolArgs)"), false);
  assert.equal(
    ADAPTER.includes("argsText: JSON.stringify(mergedArgs ?? {})"),
    false,
  );
});

test("prompt export routes through the exact-text replay helper", () => {
  assert.match(PROMPT_STORAGE, /const argsStr = toolCallReplayArguments\(/);
  assert.equal(
    PROMPT_STORAGE.includes("p.args != null ? JSON.stringify(p.args)"),
    false,
  );
});

test("argument text that does not parse is not shown on an approval card", () => {
  for (const junk of [
    "not json at all",
    "   ",
    '{"id":1',
    "<script>alert(1)</script>",
  ]) {
    assert.equal(toolCallArgumentsText(junk, { q: "gpu" }), '{"q":"gpu"}');
  }
  assert.equal(
    toolCallArgumentsText('{"id":9007199254740993}', { id: 9007199254740992 }),
    '{"id":9007199254740993}',
  );
});

test("a Gemini native_part merge keeps the executed integer", () => {
  // The adapter computes overwritten keys BEFORE folding native_part into args.google, so
  // `google` takes the lexeme-preserving path.
  const card = '{"google":{"executableCode":{"code":"print(1)"}},"id":9007199254740993}';
  const mergedArgs = {
    google: { executableCode: { code: "print(1)" }, native_part: { inlineData: "AAA" } },
    id: 9007199254740992,
  };

  const merged = mergedToolCallArgumentsText(card, mergedArgs, []);

  assert.ok(merged.includes("9007199254740993"), merged);
  assert.ok(merged.includes("native_part"), merged);
  assert.deepEqual(JSON.parse(merged), {
    google: { executableCode: { code: "print(1)" }, native_part: { inlineData: "AAA" } },
    id: 9007199254740992,
  });
});

test("a card stored before this change still merges", () => {
  const legacy = JSON.stringify({ id: 9007199254740992, q: "gpu" });
  const merged = mergedToolCallArgumentsText(legacy, { id: 9007199254740992, q: "gpu", done: true }, []);

  assert.deepEqual(JSON.parse(merged), { id: 9007199254740992, q: "gpu", done: true });
});

test("unreadable stored text falls back instead of throwing", () => {
  for (const junk of ["", "   ", "{", "not json", '{"a":1} trailing']) {
    const out = mergedToolCallArgumentsText(junk, { a: 1 }, []);
    assert.deepEqual(JSON.parse(out), { a: 1 });
  }
});
