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

test("a small page budget still yields a page that holds characters", () => {
  // Shrunk from a fuzz failure and kept verbatim: the trigger is a separator
  // sitting immediately before `end`. `pageStart` searches FORWARD from its
  // target and returns the offset just after the separator, which could reach
  // `end` itself whenever the search window was allowed to extend that far --
  // that is, whenever `maxCharacters <= BOUNDARY_SEARCH_CHARACTERS` (1,024).
  //
  // The invariant, as a sentence: a page start must stay STRICTLY BELOW the
  // page end, because `Earlier` sets the next page's end to this start. A start
  // equal to `end` yields an empty page whose own start is that same offset, so
  // the chain stops advancing and everything before it is unreachable by paging.
  const markdown = `${"x".repeat(4_000)}\n${"y".repeat(4_000)}\n`;

  for (const maxCharacters of [64, 256, 1_023, 1_024, 1_025, 2_048]) {
    const page = selectReasoningPage(markdown, {
      end: markdown.length,
      maxCharacters,
    });
    assert.ok(
      page.start < page.end,
      `start ${page.start} must stay below end ${page.end} at maxCharacters ${maxCharacters}`,
    );
    assert.equal(page.markdown.length, page.end - page.start);
    assert.ok(page.markdown.length > 0);
  }
});

test("the Earlier chain rebuilds the whole trace at every page budget", () => {
  // The property the feature rests on: paging backwards loses nothing. Run at
  // budgets on both sides of BOUNDARY_SEARCH_CHARACTERS, because the bug above
  // was invisible at the shipped 8,192 default and only appeared below it.
  const markdown = `${longReasoning}\n\`\`\`py\n${"c = 1\n".repeat(400)}\`\`\`\n${longReasoning}`;

  for (const maxCharacters of [128, 1_024, 4_096, REASONING_PAGE_CHARACTERS]) {
    const pages: string[] = [];
    let end: number | null = null;
    for (let step = 0; step < 5_000; step += 1) {
      const page = selectReasoningPage(markdown, { end, maxCharacters });
      assert.ok(page.start < page.end || page.end === 0);
      pages.push(page.markdown);
      if (!page.hasEarlier || page.start === 0) {
        break;
      }
      end = page.start;
    }
    assert.equal(pages.reverse().join(""), markdown);
  }
});

test("a page budget shorter than a surrogate pair keeps the character whole", () => {
  const markdown = `${"a".repeat(200)}\n\u{1F600}`;
  const page = selectReasoningPage(markdown, {
    end: markdown.length,
    maxCharacters: 1,
  });

  assert.ok(page.markdown.length > 0);
  // Never a lone half: the page carries the whole astral character or none of
  // it. Iterating a string yields code POINTS, so a surviving lone surrogate
  // shows up here as a single unit still inside the surrogate range.
  for (const codePoint of page.markdown) {
    const value = codePoint.codePointAt(0) ?? 0;
    assert.ok(
      value < 0xd800 || value > 0xdfff,
      `page opened on a lone surrogate (U+${value.toString(16)})`,
    );
  }
});
