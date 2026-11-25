// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { Nodes } from "mdast";
import { fromMarkdown } from "mdast-util-from-markdown";

import {
  FENCE_CACHE_MAX_CHARS,
  FENCE_CACHE_MAX_ENTRIES,
  clearFenceCache,
  collectChatFiles,
  extractCodeFences,
  fenceCacheStats,
} from "../src/features/chat/lib/chat-files.ts";

const MAY_HOLD_CODE_RE = /```|~~~|^(?: {4}|\t)/m;

// Uncached reference extraction.
function reference(markdown: string) {
  if (!MAY_HOLD_CODE_RE.test(markdown)) return [];
  const blocks: { language: string | null; source: string }[] = [];
  const visit = (node: Nodes) => {
    if (node.type === "code") {
      if (node.value.trim()) {
        blocks.push({ language: node.lang ?? null, source: node.value });
      }
      return;
    }
    if ("children" in node) node.children.forEach(visit);
  };
  visit(fromMarkdown(markdown));
  return blocks;
}

const CORPUS = [
  "Plain prose, no code at all.",
  "Here:\n```python\nprint(1)\n```\n",
  "```\nno language\n```",
  "```js\n\n   \n```\n",
  "~~~html\n<p>tilde</p>\n~~~\n",
  "~~~~\n```\nnested backticks inside tildes\n```\n~~~~\n",
  "    indented code line\n    second line\n",
  "\tTab indented code\n",
  "1. Step\n\n   ```bash\n   pip install x\n   ```\n\n2. Next\n",
  "- outer\n  - inner\n\n        indented in nested list\n\n    ```ts\n    const a = 1;\n    ```\n",
  "```python\nunterminated fence\nruns to the end\n",
  "```py\r\nprint('crlf')\r\n```\r\nafter\r\n",
  "> quoted\n> ```sh\n> echo hi\n> ```\n",
  "```a\none\n```\ntext\n```b\ntwo\n```\n~~~c\nthree\n~~~\n",
  "inline `code` and ``` not a fence on its own line",
  "* * *\n- - -\n```\nafter breaks\n```",
  "```html\n<!doctype html><html><body>x</body></html>\n```",
  " ```  spaced info string  \n body\n ```",
  "",
];

test("cached blocks equal the uncached parse on first and repeated reads", () => {
  clearFenceCache();
  for (const text of CORPUS) {
    const expected = reference(text);
    assert.deepEqual(
      extractCodeFences(text),
      expected,
      `miss: ${JSON.stringify(text)}`,
    );
    assert.deepEqual(
      extractCodeFences(text),
      expected,
      `hit: ${JSON.stringify(text)}`,
    );
  }
  const rebuilt = ["Here:\n```py", "thon\nprint(1)\n```\n"].join("");
  assert.deepEqual(extractCodeFences(rebuilt), reference(rebuilt));
});

test("a caller mutating the result does not change the cached answer", () => {
  clearFenceCache();
  const text = "```py\nx = 1\n```";
  const first = extractCodeFences(text);
  first[0].source = "changed";
  first.push({ language: null, source: "extra" });
  assert.deepEqual(extractCodeFences(text), reference(text));
});

test("text with no fence or indented line is not cached", () => {
  clearFenceCache();
  extractCodeFences("just words");
  assert.deepEqual(fenceCacheStats(), { keys: [], chars: 0 });
});

test("entry bound evicts the least recently used text first", () => {
  clearFenceCache();
  const text = (i: number) => `\`\`\`\nblock ${i}\n\`\`\``;
  for (let i = 0; i < FENCE_CACHE_MAX_ENTRIES; i += 1)
    extractCodeFences(text(i));
  assert.equal(fenceCacheStats().keys.length, FENCE_CACHE_MAX_ENTRIES);
  extractCodeFences(text(0)); // refresh the oldest
  extractCodeFences(text(FENCE_CACHE_MAX_ENTRIES));
  const { keys } = fenceCacheStats();
  assert.equal(keys.length, FENCE_CACHE_MAX_ENTRIES);
  assert.ok(keys.includes(text(0)));
  assert.ok(!keys.includes(text(1)));
  assert.equal(keys.at(-1), text(FENCE_CACHE_MAX_ENTRIES));
  assert.deepEqual(extractCodeFences(text(1)), reference(text(1)));
});

test("character bound holds and an oversized text is answered but not kept", () => {
  clearFenceCache();
  const body = "x".repeat(Math.floor(FENCE_CACHE_MAX_CHARS / 5));
  const big = (i: number) => `\`\`\`\n${i}${body}\n\`\`\``;
  for (let i = 0; i < 6; i += 1) {
    assert.deepEqual(extractCodeFences(big(i)), reference(big(i)));
    assert.ok(fenceCacheStats().chars <= FENCE_CACHE_MAX_CHARS);
  }
  assert.ok(fenceCacheStats().keys.length > 0);
  assert.ok(fenceCacheStats().keys.length < 6);
  clearFenceCache();
  const huge = `\`\`\`\n${"y".repeat(FENCE_CACHE_MAX_CHARS)}\n\`\`\``;
  assert.deepEqual(extractCodeFences(huge), reference(huge));
  assert.deepEqual(fenceCacheStats(), { keys: [], chars: 0 });
});

test("characters held match the entries after evictions", () => {
  clearFenceCache();
  for (let i = 0; i < FENCE_CACHE_MAX_ENTRIES + 50; i += 1) {
    extractCodeFences(`\`\`\`ts\nconst v${i} = ${"z".repeat(i % 97)};\n\`\`\``);
  }
  const { keys, chars } = fenceCacheStats();
  assert.equal(keys.length, FENCE_CACHE_MAX_ENTRIES);
  const expected = keys.reduce(
    (sum, key) =>
      sum +
      key.length +
      reference(key).reduce(
        (s, b) => s + b.source.length + (b.language?.length ?? 0),
        0,
      ),
    0,
  );
  assert.equal(chars, expected);
});

test("a reopened chat (new message objects, same texts) lists the same files", () => {
  clearFenceCache();
  const thread = () =>
    CORPUS.map((text, i) => ({
      id: `a${i}`,
      role: "assistant",
      content: [{ type: "text", text }],
    }));
  const first = collectChatFiles(thread());
  const held = fenceCacheStats();
  assert.deepEqual(
    held.keys,
    CORPUS.filter((text) => MAY_HOLD_CODE_RE.test(text)),
  );
  const reopened = collectChatFiles(thread());
  assert.deepEqual(reopened, first);
  assert.deepEqual(fenceCacheStats(), held);
  clearFenceCache();
  assert.deepEqual(collectChatFiles(thread()), first);
  assert.ok(first.length > 10);
});
