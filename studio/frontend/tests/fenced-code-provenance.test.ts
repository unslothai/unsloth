// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { register } from "node:module";
import test from "node:test";

import {
  IncrementalMarkdownCache,
  type IncrementalMarkdownCodeFence,
  type IncrementalMarkdownRender,
} from "../src/components/assistant-ui/streaming-render-schedule.ts";
import {
  MAX_AUTO_HIGHLIGHT_SOURCE_CODE_UNITS,
  shouldAutoHighlightStreamingCode,
} from "../src/components/assistant-ui/streaming-code-policy.ts";
import {
  AssistantFencedCodeProvenanceTracker,
  FENCED_CODE_PROVENANCE_KEY,
  getCompletedFencedCodeOccurrences,
  readFencedCodeProvenance,
} from "../src/lib/fenced-code-provenance.ts";

register("./helpers/delete-thread-message-resolver.mjs", import.meta.url);
const { exportedItemToRecord } = await import(
  "../src/features/chat/utils/delete-thread-message.ts"
);
const { updateThreadMessage } = await import(
  "../src/features/chat/utils/update-thread-message.ts"
);

type TestPart = {
  type: string;
  text?: string;
  [FENCED_CODE_PROVENANCE_KEY]?: unknown;
};

const provenance = (part: TestPart | undefined): unknown =>
  part?.[FENCED_CODE_PROVENANCE_KEY];

const codeFences = (
  render: IncrementalMarkdownRender,
): IncrementalMarkdownCodeFence[] => [
  ...render.chunks.flatMap((chunk) =>
    chunk.blocks.flatMap((block) => block.codeFences),
  ),
  ...render.tail.flatMap((block) => block.codeFences),
];

const hash = (value: string): string =>
  createHash("sha256").update(value).digest("hex");

const sourceOfLength = (length: number): string => {
  assert.ok(length >= 3);
  const body = "const value = '💡';\n".repeat(Math.ceil(length / 20));
  const prefix = body.slice(0, length - 1);
  return `${prefix}\n`;
};

test("published boundaries annotate only the text part that owned the closing separator", () => {
  const tracker = new AssistantFencedCodeProvenanceTracker();
  const open = "```ts\nconst value = 1;\n";
  const reasoning = { type: "reasoning", text: open };

  const beforeClose = tracker.annotate<TestPart>([
    reasoning,
    { type: "text", text: open },
  ]);
  assert.equal(beforeClose[0], reasoning);
  assert.equal(provenance(beforeClose[0]), undefined);
  assert.equal(provenance(beforeClose[1]), undefined);

  const closed = tracker.annotate<TestPart>([
    reasoning,
    { type: "text", text: `${open}\`\`\`` },
  ]);
  assert.deepEqual(provenance(closed[1]), { v: 1, trailingLf: [0] });
  assert.equal(provenance(closed[0]), undefined);

  const followedByProse = tracker.annotate<TestPart>([
    reasoning,
    { type: "text", text: `${open}\`\`\`\n\nDone.` },
  ]);
  assert.deepEqual(provenance(followedByProse[1]), {
    v: 1,
    trailingLf: [0],
  });
});

test("a separator never published at the live edge is not guessed", () => {
  const noFinalLf = new AssistantFencedCodeProvenanceTracker();
  noFinalLf.annotate<TestPart>([{ type: "text", text: "```js\nvalue" }]);
  const closedInOnePublish = noFinalLf.annotate<TestPart>([
    { type: "text", text: "```js\nvalue\n```" },
  ]);
  assert.equal(provenance(closedInOnePublish[0]), undefined);

  const withheld = new AssistantFencedCodeProvenanceTracker();
  withheld.annotate<TestPart>([{ type: "text", text: "```js\nvalue" }]);
  // The publish gate withheld the snapshot ending in LF, so the tracker must not
  // learn from it merely because the same bytes appear in a later close.
  const closeAfterWithheldSnapshot = withheld.annotate<TestPart>([
    { type: "text", text: "```js\nvalue\n```" },
  ]);
  assert.equal(provenance(closeAfterWithheldSnapshot[0]), undefined);
});

test("duplicates, mixed text parts, CRLF, and lone CR stay part-local", () => {
  for (const lineEnding of ["\n", "\r\n", "\r"]) {
    const tracker = new AssistantFencedCodeProvenanceTracker();
    const firstOpen = `~~~js${lineEnding}same${lineEnding}`;
    tracker.annotate<TestPart>([{ type: "text", text: firstOpen }]);
    const secondOpen = `${firstOpen}~~~${lineEnding}${lineEnding}\`\`\`js${lineEnding}same${lineEnding}`;
    const between = tracker.annotate<TestPart>([
      { type: "text", text: secondOpen },
      { type: "tool-call", text: firstOpen },
      { type: "text", text: `\`\`\`txt${lineEnding}other${lineEnding}` },
    ]);
    assert.deepEqual(provenance(between[0]), { v: 1, trailingLf: [0] });
    assert.equal(provenance(between[1]), undefined);
    assert.equal(provenance(between[2]), undefined);

    const closed = tracker.annotate<TestPart>([
      { type: "text", text: `${secondOpen}\`\`\`` },
      { type: "tool-call", text: firstOpen },
      {
        type: "text",
        text: `\`\`\`txt${lineEnding}other${lineEnding}\`\`\``,
      },
    ]);
    assert.deepEqual(provenance(closed[0]), { v: 1, trailingLf: [0, 1] });
    assert.deepEqual(provenance(closed[2]), { v: 1, trailingLf: [0] });
  }
});

test("rewinds discard stale ownership and rescan only the changed suffix", () => {
  const tracker = new AssistantFencedCodeProvenanceTracker();
  const first = "prefix\n\n```ts\nold value\n";
  tracker.annotate<TestPart>([{ type: "text", text: first }]);
  const firstStats = tracker.stats();
  assert.equal(firstStats.scannedCharacters, first.length);

  const replacement = "prefix\n\n```ts\nnew value\n";
  tracker.annotate<TestPart>([{ type: "text", text: replacement }]);
  const closed = tracker.annotate<TestPart>([
    { type: "text", text: `${replacement}\`\`\`` },
  ]);
  assert.deepEqual(provenance(closed[0]), { v: 1, trailingLf: [0] });
  assert.equal(tracker.stats().rewinds, 1);

  const removed = tracker.annotate<TestPart>([
    {
      type: "text",
      text: "prefix\n\nplain replacement",
      [FENCED_CODE_PROVENANCE_KEY]: { v: 1, trailingLf: [0] },
    },
  ]);
  assert.equal(provenance(removed[0]), undefined);
});

test("ordinary append tracking scans each canonical character once", () => {
  const tracker = new AssistantFencedCodeProvenanceTracker();
  let text = `\`\`\`text\n${"x".repeat(80_000)}`;
  tracker.annotate<TestPart>([{ type: "text", text }]);
  for (const delta of ["\n", "`", "`", "`"]) {
    text += delta;
    tracker.annotate<TestPart>([{ type: "text", text }]);
  }
  const once = tracker.stats();
  assert.equal(once.scannedCharacters, text.length);
  assert.equal(once.rewinds, 0);
  tracker.annotate<TestPart>([{ type: "text", text }]);
  assert.deepEqual(tracker.stats(), once);
});

test("the persisted extension reader rejects every malformed shape", () => {
  assert.deepEqual(
    readFencedCodeProvenance({
      type: "text",
      text: "",
      [FENCED_CODE_PROVENANCE_KEY]: { v: 1, trailingLf: [0, 2, 9] },
    }),
    [0, 2, 9],
  );

  for (const malformed of [
    null,
    [],
    { v: 2, trailingLf: [0] },
    { v: 1, trailingLf: "0" },
    { v: 1, trailingLf: [-1] },
    { v: 1, trailingLf: [1, 1] },
    { v: 1, trailingLf: [2, 1] },
    { v: 1, trailingLf: [0.5] },
    { v: 1, trailingLf: [Number.MAX_SAFE_INTEGER + 1] },
    { v: 1, trailingLf: [0], extra: true },
  ]) {
    assert.deepEqual(
      readFencedCodeProvenance({
        type: "text",
        text: "",
        [FENCED_CODE_PROVENANCE_KEY]: malformed,
      }),
      [],
      JSON.stringify(malformed),
    );
  }
});

test("the line scanner records completed CommonMark fence ordinals and separator evidence", () => {
  const markdown = [
    "```js",
    "same",
    "```",
    "",
    "~~~txt",
    "same",
    "~~~~  ",
    "",
    "```bad`info",
    "not a fence",
  ].join("\n");
  const occurrences = getCompletedFencedCodeOccurrences(markdown);
  assert.deepEqual(
    occurrences.map(({ ordinal, bodyWithSeparator }) => ({
      ordinal,
      bodyWithSeparator,
    })),
    [
      { ordinal: 0, bodyWithSeparator: "same\n" },
      { ordinal: 1, bodyWithSeparator: "same\n" },
    ],
  );
});

test("cold provenance restores the exact 16,384/16,385-unit source and policy", () => {
  for (const target of [
    MAX_AUTO_HIGHLIGHT_SOURCE_CODE_UNITS,
    MAX_AUTO_HIGHLIGHT_SOURCE_CODE_UNITS + 1,
  ]) {
    const source = sourceOfLength(target);
    assert.equal(source.length, target);
    const markdown = `\`\`\`typescript\n${source}\`\`\``;
    const legacyFence = codeFences(
      new IncrementalMarkdownCache().update(markdown, false),
    )[0];
    assert.equal(legacyFence.source.length, target - 1);

    const restoredFence = codeFences(
      new IncrementalMarkdownCache([0]).update(markdown, false),
    )[0];
    assert.equal(restoredFence.source.length, target);
    assert.equal(restoredFence.source, source);
    assert.equal(hash(restoredFence.source), hash(source));
    assert.equal(
      shouldAutoHighlightStreamingCode(restoredFence.source),
      target <= MAX_AUTO_HIGHLIGHT_SOURCE_CODE_UNITS,
    );
  }
});


test("same-text live provenance immediately matches a fresh cold cache", () => {
  const source = sourceOfLength(5_954);
  const prefix = Array.from(
    { length: 80 },
    (_, index) => `Stable paragraph ${index}.\n\n`,
  ).join("");
  const markdown = `${prefix}\`\`\`typescript\n${source}\`\`\``;
  const cache = new IncrementalMarkdownCache();
  const before = cache.update(markdown, false, []);
  const beforeFence = codeFences(before)[0];
  const unaffectedChunk = before.chunks[0];
  const identityAfterParse = (
    cache as unknown as { nextBlockIdentity: number }
  ).nextBlockIdentity;
  assert.ok(unaffectedChunk);
  assert.equal(beforeFence.source.length, 5_953);

  const live = cache.update(markdown, false, [0]);
  const liveFence = codeFences(live)[0];
  const cold = new IncrementalMarkdownCache([0]).update(markdown, false);
  const coldFence = codeFences(cold)[0];

  assert.notEqual(live, before, "changed metadata must revise the live plan");
  assert.equal(live.chunks[0], unaffectedChunk);
  assert.equal(
    (cache as unknown as { nextBlockIdentity: number }).nextBlockIdentity,
    identityAfterParse,
    "a provenance-only revision reparsed unchanged Markdown",
  );
  assert.equal(liveFence.source.length, 5_954);
  assert.equal(liveFence.source, coldFence.source);
  assert.equal(hash(liveFence.source), hash(coldFence.source));
  assert.equal(
    cache.update(markdown, false, [0]),
    live,
    "repeated metadata must keep the exact plan identity",
  );

  const removed = cache.update(markdown, false, []);
  assert.equal(codeFences(removed)[0].source, beforeFence.source);
  assert.equal(removed.chunks[0], unaffectedChunk);
  assert.equal(cache.update(markdown, false, []), removed);
});

test("CRLF cold reload normalizes bytes once and preserves a surrogate boundary", () => {
  const source = sourceOfLength(MAX_AUTO_HIGHLIGHT_SOURCE_CODE_UNITS + 1);
  assert.match(source, /💡/u);
  const rawMarkdown = `~~~~typescript\r\n${source.replaceAll("\n", "\r\n")}~~~~`;
  const restored = codeFences(
    new IncrementalMarkdownCache([0]).update(rawMarkdown, false),
  )[0];
  assert.equal(restored.source, source);
  assert.equal(restored.source.length, MAX_AUTO_HIGHLIGHT_SOURCE_CODE_UNITS + 1);
  assert.equal(shouldAutoHighlightStreamingCode(restored.source), false);
});

test("cold overlay spans committed blocks, duplicates, prose, and global footnotes", () => {
  const prefix = Array.from(
    { length: 80 },
    (_, index) => `Paragraph ${index}.\n\n`,
  ).join("");
  const first = "first source\n";
  const second = "first source\n";
  const markdown = `${prefix}\`\`\`txt\n${first}\`\`\`\n\nA note[^n].\n\n~~~txt\n${second}~~~\n\n[^n]: global detail`;
  const render = new IncrementalMarkdownCache([0, 1]).update(markdown, false);
  assert.deepEqual(
    codeFences(render).map((fence) => fence.source),
    [first, second],
  );
});

test("out-of-range provenance is all-or-nothing and legacy content stays canonical", () => {
  const source = "value\n";
  const markdown = `\`\`\`txt\n${source}\`\`\``;
  const legacy = codeFences(
    new IncrementalMarkdownCache().update(markdown, false),
  )[0];
  const malformed = codeFences(
    new IncrementalMarkdownCache([0, 9]).update(markdown, false),
  )[0];
  assert.equal(legacy.source, "value");
  assert.equal(malformed.source, legacy.source);
});

test("frontend autosave/deep clone preserve provenance while an edit drops it", async () => {
  const annotatedPart = {
    type: "text" as const,
    text: "```text\nvalue\n```",
    [FENCED_CODE_PROVENANCE_KEY]: { v: 1, trailingLf: [0] },
  };
  const message = {
    id: "assistant-1",
    role: "assistant" as const,
    content: [
      annotatedPart,
      {
        type: "source" as const,
        sourceType: "url" as const,
        id: "source-1",
        url: "https://example.com",
      },
    ],
    createdAt: new Date(1),
    status: { type: "complete" as const, reason: "stop" as const },
    metadata: {
      custom: {},
      unstable_state: null,
      unstable_annotations: [],
      unstable_data: [],
      steps: [],
    },
  };
  const record = exportedItemToRecord("thread-1", null, message);
  assert.deepEqual(record.content, message.content);
  const cloned = JSON.parse(JSON.stringify(record.content));
  assert.deepEqual(cloned, message.content);
  assert.notEqual(cloned[0], annotatedPart);

  const initial = {
    headId: message.id,
    messages: [{ parentId: null, message }],
  };
  let imported = initial;
  await updateThreadMessage({
    thread: {
      export: () => initial,
      import: (next) => {
        imported = next as typeof initial;
      },
    },
    messageId: message.id,
    remoteId: undefined,
    newText: "edited text",
    isIncognito: false,
  });
  const editedContent = imported.messages[0].message.content;
  assert.deepEqual(editedContent[0], { type: "text", text: "edited text" });
  assert.equal(
    (editedContent[0] as Record<string, unknown>)[FENCED_CODE_PROVENANCE_KEY],
    undefined,
  );
  assert.deepEqual(editedContent[1], message.content[1]);
});


test("every assistant yield is decorated at its publication boundary", () => {
  const source = readFileSync(
    new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
    "utf8",
  );
  const yields = [...source.matchAll(/\byield\s*\{/g)];
  assert.equal(yields.length, 11);
  for (const match of yields) {
    const excerpt = source.slice(match.index, match.index + 500);
    assert.match(
      excerpt,
      /content:\s*publishedAssistantContent\(/,
      `unwrapped assistant yield near ${match.index}`,
    );
  }
  assert.doesNotMatch(source, /yield\s*\{[\s\S]{0,200}content:\s*liveAssistantContent\(/);
});
