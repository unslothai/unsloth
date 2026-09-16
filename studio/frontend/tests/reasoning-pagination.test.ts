// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  REASONING_PAGE_CHARACTERS,
  REASONING_PAGINATION_THRESHOLD,
  ReasoningPageSelector,
  createReasoningPageBoundary,
  isReasoningPageBoundaryValid,
  selectReasoningPage,
  shouldPaginateReasoning,
} from "../src/components/assistant-ui/reasoning-pagination.ts";

const longReasoning = Array.from(
  { length: 2_000 },
  (_, index) => `paragraph ${index}: ${"reasoning ".repeat(8)}\n`,
).join("\n");

test("ordinary short reasoning is unchanged and does not paginate", () => {
  const markdown = "# Thought\n\nA short explanation with `code`.";
  const page = selectReasoningPage(markdown);

  assert.equal(shouldPaginateReasoning(markdown), false);
  assert.deepEqual(page, {
    end: markdown.length,
    hasEarlier: false,
    hasNewer: false,
    markdown,
    oversizedCode: false,
    start: 0,
  });
  assert.equal(
    shouldPaginateReasoning("x".repeat(REASONING_PAGINATION_THRESHOLD)),
    false,
  );
  assert.equal(
    shouldPaginateReasoning("x".repeat(REASONING_PAGINATION_THRESHOLD + 1)),
    true,
  );
});

test("long reasoning mounts one bounded page and navigation loses no source", () => {
  const newestFirst = [selectReasoningPage(longReasoning)];
  while (newestFirst.at(-1)?.hasEarlier) {
    newestFirst.push(
      selectReasoningPage(longReasoning, {
        end: newestFirst.at(-1)?.start,
      }),
    );
  }
  const pages = newestFirst.reverse();

  assert.ok(pages.length > 10);
  assert.ok(
    pages.every((page) => page.markdown.length <= REASONING_PAGE_CHARACTERS),
  );
  assert.equal(pages.map((page) => page.markdown).join(""), longReasoning);
  assert.equal(pages[0].hasEarlier, false);
  assert.equal(pages.at(-1)?.hasNewer, false);
  assert.ok(pages.slice(0, -1).every((page) => page.hasNewer));
});

test("an earlier page and its boundary stay stable while the live tail grows", () => {
  const latest = selectReasoningPage(longReasoning);
  const boundary = createReasoningPageBoundary(longReasoning, latest.start);
  const earlier = selectReasoningPage(longReasoning, { end: boundary.end });
  const appended = `${longReasoning}\nnew streamed tokens`;

  assert.equal(isReasoningPageBoundaryValid(appended, boundary), true);
  assert.deepEqual(
    selectReasoningPage(appended, { end: boundary.end }),
    earlier,
  );

  const rewritten = `${longReasoning.slice(0, boundary.end - 2)}XX${longReasoning.slice(boundary.end)}`;
  assert.equal(isReasoningPageBoundaryValid(rewritten, boundary), false);
});

test("giant closed and open fences use bounded visibly-safe fallback pages", () => {
  for (const markdown of [
    `\`\`\`typescript\n${"const value = 1;\n".repeat(2_000)}\`\`\`\n`,
    `~~~python\n${"print('streaming')\n".repeat(2_000)}`,
  ]) {
    const latest = selectReasoningPage(markdown);
    const earlier = selectReasoningPage(markdown, { end: latest.start });

    for (const page of [latest, earlier]) {
      assert.ok(page.markdown.length <= REASONING_PAGE_CHARACTERS);
      assert.equal(page.oversizedCode, true);
    }
  }
});

test("the streaming selector incrementally tracks append and resets on replacement", () => {
  const selector = new ReasoningPageSelector();
  const openFence = `prefix\n\n~~~ts\n${"const x = 1;\n".repeat(1_000)}`;
  const appended = `${openFence}~~~\n\nlatest paragraph\n`;

  assert.deepEqual(selector.select(openFence), selectReasoningPage(openFence));
  assert.deepEqual(selector.select(appended), selectReasoningPage(appended));
  assert.deepEqual(
    selector.select(longReasoning),
    selectReasoningPage(longReasoning),
  );
});
test("live pages advance in stable half-page strides", () => {
  const selector = new ReasoningPageSelector();
  const maxCharacters = 4_096;
  let markdown = longReasoning.slice(0, 20_000);
  const first = selector.select(markdown, { maxCharacters, streaming: true });

  markdown += "streamed tail ".repeat(500);
  const advanced = selector.select(markdown, {
    maxCharacters,
    streaming: true,
  });
  const room = maxCharacters - advanced.markdown.length;
  assert.ok(advanced.start > first.start);
  assert.ok(room > 100);

  markdown += "x".repeat(Math.floor(room / 2));
  const stable = selector.select(markdown, { maxCharacters, streaming: true });
  assert.equal(stable.start, advanced.start);
  assert.ok(stable.markdown.length <= maxCharacters);

  markdown += "y".repeat(room);
  const nextStride = selector.select(markdown, {
    maxCharacters,
    streaming: true,
  });
  assert.ok(nextStride.start > stable.start);
  assert.ok(nextStride.markdown.length <= maxCharacters);
});

test("same-edge equal-length replacement resets incremental fence state", () => {
  const selector = new ReasoningPageSelector();
  const edge = "edge".repeat(512);
  const fence = `\n\`\`\`ts\n${"const stale = true;\n".repeat(600)}\`\`\`\n`;
  const oldSource = edge + fence + edge;
  const replacement = edge + " ".repeat(fence.length) + edge;

  assert.equal(
    selector.select(oldSource, { maxCharacters: 4_096 }).oversizedCode,
    true,
  );
  assert.deepEqual(
    selector.select(replacement, { maxCharacters: 4_096 }),
    selectReasoningPage(replacement, { maxCharacters: 4_096 }),
  );
  assert.equal(
    selector.select(replacement, { maxCharacters: 4_096 }).oversizedCode,
    false,
  );
});

test("CRLF bytes and unicode survive complete pagination", () => {
  const markdown = Array.from(
    { length: 1_500 },
    (_, index) => `row ${index}: 😀 café`,
  ).join("\r\n");
  const newestFirst = [selectReasoningPage(markdown, { maxCharacters: 1_024 })];
  while (newestFirst.at(-1)?.hasEarlier) {
    newestFirst.push(
      selectReasoningPage(markdown, {
        end: newestFirst.at(-1)?.start,
        maxCharacters: 1_024,
      }),
    );
  }
  const rebuilt = newestFirst
    .reverse()
    .map((page) => page.markdown)
    .join("");

  assert.equal(rebuilt, markdown);
  assert.equal(rebuilt.includes("\r\n"), true);
  assert.equal(rebuilt.includes("😀"), true);
  assert.equal(rebuilt.includes("\uFFFD"), false);
});
