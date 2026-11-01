// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

register("./chat-search-index-resolver.mjs", import.meta.url);
installLocalStorageFake();

const { collapseWhitespace, stripBase64Runs } = await import(
  "../src/features/chat/utils/chat-search-text.ts"
);
const { buildChatSearchIndex } = await import(
  "../src/features/chat/hooks/use-chat-search-index.ts"
);
const { chatSearchBatchRequests, configureChatSearchHistoryStub } =
  await import("./helpers/store-stubs/chat-search-history.ts");

function randomStrings(alphabet: string[], count: number): string[] {
  let seed = 12345;
  const next = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed;
  };
  const out: string[] = [];
  for (let i = 0; i < count; i++) {
    let s = "";
    const len = next() % 400;
    for (let j = 0; j < len; j++) {
      // Long runs of one class, so 120+ runs and near misses both occur.
      const piece = alphabet[next() % alphabet.length];
      s += next() % 4 === 0 ? piece.repeat(1 + (next() % 130)) : piece;
    }
    out.push(s);
  }
  return out;
}

test("stripBase64Runs matches the regex it replaces", () => {
  const regex = /[A-Za-z0-9+/]{120,}={0,2}/g;
  const fixed = [
    "",
    "a".repeat(119),
    "a".repeat(120),
    `x ${"A".repeat(120)}=== y`,
    `${"b/+".repeat(40)}==`,
    `${"Q".repeat(200)}=${"Q".repeat(200)}`,
    `data ${"9".repeat(500)} tail`,
    "\u00e9".repeat(200),
  ];
  const fuzz = randomStrings(
    ["A", "z", "0", "+", "/", "=", " ", "-", "\u00e9", "\n", "ab", "=="],
    3000,
  );
  for (const s of [...fixed, ...fuzz]) {
    assert.equal(stripBase64Runs(s), s.replace(regex, " "), JSON.stringify(s));
  }
});

test("collapseWhitespace matches the regex it replaces", () => {
  const regex = /\s+/g;
  const fixed = [
    "",
    " ",
    "  ",
    " a ",
    "a\tb",
    "a \n b",
    "\u00a0",
    " \ufeff",
    "x\u3000 y",
    "\u2028",
  ];
  const fuzz = randomStrings(
    [" ", "\t", "\n", "\r", "\u00a0", "\u2028", "\ufeff", "\u3000", "a", "bc"],
    3000,
  );
  for (const s of [...fixed, ...fuzz]) {
    assert.equal(
      collapseWhitespace(s),
      s.replace(regex, " "),
      JSON.stringify(s),
    );
  }
});

test("the search index asks the backend for the search view", async () => {
  configureChatSearchHistoryStub({
    threads: [{ id: "t1", title: "Acme", createdAt: 1 }],
    messagesByThread: new Map([
      [
        "t1",
        [
          {
            id: "m1",
            threadId: "t1",
            role: "user",
            content: [{ type: "text", text: "Roadmap  notes" }],
            createdAt: 2,
          },
        ],
      ],
    ]),
  });
  const build = await buildChatSearchIndex();
  assert.deepEqual(chatSearchBatchRequests(), [
    { threadIds: ["t1"], options: { view: "search" } },
  ]);
  assert.equal(build.items[0].searchText, "acme roadmap notes");
});

test("the backend search view keeps every part type the index reads", () => {
  const hook = readFileSync(
    new URL(
      "../src/features/chat/hooks/use-chat-search-index.ts",
      import.meta.url,
    ),
    "utf8",
  );
  const start = hook.indexOf("function extractText(");
  const body = hook.slice(start, hook.indexOf("\n}\n", start));
  const read = new Set(
    [...body.matchAll(/p\.type === "([^"]+)"/g)].map((m) => m[1]),
  );
  const route = readFileSync(
    new URL("../../backend/routes/chat_history.py", import.meta.url),
    "utf8",
  );
  const kept = /_SEARCH_PART_TYPES = frozenset\(\(([^)]*)\)\)/.exec(route)?.[1];
  assert.ok(kept, "routes/chat_history.py must define _SEARCH_PART_TYPES");
  const keptTypes = new Set([...kept.matchAll(/"([^"]+)"/g)].map((m) => m[1]));
  assert.deepEqual([...keptTypes].sort(), [...read].sort());
});
