import assert from "node:assert/strict";
import test from "node:test";

import type {
  MessageRecord,
  ThreadRecord,
} from "../src/features/chat/types.ts";
import {
  fallbackTitleFromUserText,
  isLegacyClippedTitle,
  planLegacyTitleRepairs,
  selectLegacyRepairPage,
  threadsMissingMessages,
} from "../src/features/chat/utils/chat-title.ts";

const LONG =
  "Can you plot a Mandelbrot set and explain how the escape time algorithm works";

function thread(id: string, title: string): ThreadRecord {
  return { id, title, createdAt: 1, updatedAt: 1 } as ThreadRecord;
}

function userMessage(threadId: string, text: string): MessageRecord {
  return {
    id: `${threadId}-m1`,
    threadId,
    role: "user",
    content: [{ type: "text", text }],
    createdAt: 1,
  } as MessageRecord;
}

test("a title the sidebar can clip keeps the whole first line", () => {
  assert.equal(fallbackTitleFromUserText(LONG), LONG);
  assert.equal(fallbackTitleFromUserText("  spaced   out  "), "spaced out");
  assert.equal(fallbackTitleFromUserText("first\nsecond"), "first");
  assert.equal(fallbackTitleFromUserText("   "), "New Chat");
});

test("only a pasted wall of text is cut, and with a real ellipsis", () => {
  const wall = "x".repeat(200);
  const title = fallbackTitleFromUserText(wall);
  assert.equal(title.length, 121);
  assert.ok(title.endsWith("…"));
  assert.ok(!title.includes("..."));
});

/** A high surrogate with no low after it, or a low with no high before it. */
const UNPAIRED_SURROGATE =
  /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/;

test("the cap never splits an emoji into a lone surrogate", () => {
  // A lone surrogate survives JSON.stringify but fails the backend's SQLite
  // bind, so the title write would 500.
  const line = "x".repeat(119) + "\u{1F600} tail";
  assert.equal(UNPAIRED_SURROGATE.test(line.slice(0, 120)), true);
  const title = fallbackTitleFromUserText(line);
  assert.equal(UNPAIRED_SURROGATE.test(title), false);
  assert.equal(title, "x".repeat(119) + "\u{1F600}" + "…");
});

test("a legacy title is recognised only against the text it was cut from", () => {
  const legacy = LONG.slice(0, 48) + "...";
  assert.equal(isLegacyClippedTitle(legacy, LONG), true);
  assert.equal(
    isLegacyClippedTitle(legacy, "a different first message"),
    false,
  );
  // A rename that merely ends in "..." is left alone.
  assert.equal(isLegacyClippedTitle("Wait for it...", LONG), false);
  assert.equal(isLegacyClippedTitle(LONG, LONG), false);
});

test("repair rewrites legacy rows and leaves every other row untouched", () => {
  const legacy = LONG.slice(0, 48) + "...";
  const threads = [
    thread("a", legacy),
    thread("b", "Mandelbrot escape time"),
    thread("c", legacy),
  ];
  const messages = new Map<string, MessageRecord[]>([
    ["a", [userMessage("a", LONG)]],
    ["b", [userMessage("b", LONG)]],
    // No stored messages: nothing to rewrite the title from.
    ["c", []],
  ]);

  assert.deepEqual(planLegacyTitleRepairs(threads, messages), [
    { threadId: "a", previousTitle: legacy, title: LONG },
  ]);
});

test("a page skips rows already tried and reports the leftovers", () => {
  const legacy = LONG.slice(0, 48) + "...";
  const threads = [
    thread("a", legacy),
    thread("b", "a plain title"),
    thread("c", legacy),
    thread("d", legacy),
  ];

  const first = selectLegacyRepairPage(threads, new Set(), 2);
  assert.deepEqual(
    first.candidates.map((t) => t.id),
    ["a", "c"],
  );
  // Without this the rest of a long history waits on an unrelated refresh.
  assert.equal(first.hasMore, true);

  const second = selectLegacyRepairPage(threads, new Set(["a", "c"]), 2);
  assert.deepEqual(
    second.candidates.map((t) => t.id),
    ["d"],
  );
  assert.equal(second.hasMore, false);

  const done = selectLegacyRepairPage(threads, new Set(["a", "c", "d"]), 2);
  assert.deepEqual(done.candidates, []);
  assert.equal(done.hasMore, false);
});

test("a thread the backend has nothing for still gets a local read", () => {
  // A legacy chat not yet imported reads empty from the backend, and an id it
  // has never heard of is missing from the map entirely.
  const messages = new Map<string, MessageRecord[]>([
    ["a", [userMessage("a", LONG)]],
    ["b", []],
  ]);
  assert.deepEqual(threadsMissingMessages(["a", "b", "c"], messages), [
    "b",
    "c",
  ]);
});
