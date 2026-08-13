// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import remend from "remend";
import { parseMarkdownIntoBlocks } from "streamdown";

import {
  IncrementalMarkdownCache,
  withoutStreamdownAnimationPlugin,
} from "../src/components/assistant-ui/streaming-render-schedule.ts";

test("only Streamdown's animation transformer is removed", () => {
  const first = () => undefined;
  const animation = () => undefined;
  const configured: [typeof first, { enabled: boolean }] = [
    first,
    {
      enabled: true,
    },
  ];
  const plugins = [first, animation, configured];

  assert.deepEqual(
    withoutStreamdownAnimationPlugin(plugins, {
      name: "animate",
      type: "animate",
      rehypePlugin: animation,
      getLastRenderCharCount: () => 0,
      setPrevContentLength: () => undefined,
    }),
    [first, configured],
  );
});

const MARKDOWN_CASES = [
  "# Heading\n\nParagraph with **bold** and [link](https://x.test).\n\n- one\n- two\n\nDone.",
  "Before\n\n```ts\nconst x = 1;\n```\n\nAfter",
  "Math \\(x+1\\)\n\n\\[a=b\\]\n\nend",
  "> quote\n> next\n\nparagraph\n\n---\n\nend",
  "<div>\ninside\n\nmore\n</div>\n\nafter",
  "A [ref][x]\n\n[x]: https://example.com\n",
  "one\n\n$$\na+b\n$$\n\ntwo",
  "a\n\n* item\n  * nested\n\nb",
  Array.from({ length: 30 }, (_, index) => `part ${index}\n\n`).join(""),
  `${Array.from({ length: 20 }, (_, index) => `part ${index}\n\n`).join("")}term[^note]\n\n[^note]: detail`,
  `$$\nx+y\n$$\n\n${Array.from({ length: 20 }, (_, index) => `part ${index}\n\n`).join("")}$$\nz`,
];

test("incremental blocks match a full Streamdown split at every prefix", () => {
  for (const source of MARKDOWN_CASES) {
    const cache = new IncrementalMarkdownCache();
    for (let length = 0; length <= source.length; length += 1) {
      const input = source.slice(0, length);
      const render = cache.update(input);
      assert.deepEqual(
        render.parseMarkdownIntoBlocks(render.markdown),
        parseMarkdownIntoBlocks(remend(input)),
        `block mismatch at prefix ${length} of ${JSON.stringify(source)}`,
      );
    }
  }
});

test("incremental parsing bounds the live tail and resets after an edit", () => {
  const source = Array.from(
    { length: 100 },
    (_, index) => `paragraph ${index}\n\n`,
  ).join("");
  const cache = new IncrementalMarkdownCache();
  const streamed = cache.update(source);
  assert.ok(streamed.markdown.length < source.length / 4);

  const edited = source.replace("paragraph 0", "changed 0");
  const reset = cache.update(edited);
  assert.deepEqual(
    reset.parseMarkdownIntoBlocks(reset.markdown),
    parseMarkdownIntoBlocks(remend(edited)),
  );
});
