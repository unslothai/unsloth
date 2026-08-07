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
  repairsStillValid,
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

/** A high surrogate with no low after it, or a low with no high before it. */
const UNPAIRED_SURROGATE =
  /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/;

test("a title the sidebar can clip keeps the whole first line", () => {
  assert.equal(fallbackTitleFromUserText(LONG), LONG);
  assert.equal(fallbackTitleFromUserText("  spaced   out  "), "spaced out");
  assert.equal(fallbackTitleFromUserText("first\nsecond"), "first");
  assert.equal(fallbackTitleFromUserText("   "), "New Chat");
});

test("only a pasted wall of text is cut, and with a real ellipsis", () => {
  const wall = "x".repeat(200);
  const title = fallbackTitleFromUserText(wall);
  // 120 UTF-16 units all in, the ellipsis included, which is what the rename
  // input accepts. A longer one cannot be edited until a character is deleted.
  assert.equal(title.length, 120);
  assert.ok(title.endsWith("…"));
  assert.ok(!title.includes("..."));
});

test("an emoji wall is capped by the same budget the input counts", () => {
  // maxLength counts UTF-16 units, so 120 astral code points would be 240.
  const title = fallbackTitleFromUserText("\u{1F600}".repeat(200));
  assert.ok(title.length <= 120);
  assert.equal(UNPAIRED_SURROGATE.test(title), false);
  assert.ok(title.endsWith("…"));
});

test("a line already inside the budget is stored whole", () => {
  const exact = "y".repeat(120);
  assert.equal(fallbackTitleFromUserText(exact), exact);
});

test("the cap never splits an emoji into a lone surrogate", () => {
  // A lone surrogate survives JSON.stringify but fails the backend's SQLite
  // bind, so the title write would 500.
  const line = "x".repeat(119) + "\u{1F600} tail";
  // A raw cut at the budget lands mid-pair.
  assert.equal(UNPAIRED_SURROGATE.test(line.slice(0, 120)), true);
  const title = fallbackTitleFromUserText(line);
  assert.equal(UNPAIRED_SURROGATE.test(title), false);
  // The emoji needs two units and only one is left, so it is left out whole.
  assert.equal(title, "x".repeat(119) + "…");
  assert.equal(title.length, 120);
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

test("a drain advances even when a whole page failed and was unmarked", () => {
  // Failures get unmarked so a later refresh can retry them. If the next page
  // were selected off the same list it would draw them straight back in and
  // spin on the failing rows, never reaching the rest.
  const legacy = LONG.slice(0, 48) + "...";
  const threads = ["a", "b", "c", "d"].map((id) => thread(id, legacy));

  const first = selectLegacyRepairPage(threads, new Set(), 2);
  assert.deepEqual(
    first.candidates.map((t) => t.id),
    ["a", "b"],
  );
  // Every write failed, so nothing stayed marked.
  const second = selectLegacyRepairPage(first.rest, new Set(), 2);
  assert.deepEqual(
    second.candidates.map((t) => t.id),
    ["c", "d"],
  );
  assert.equal(second.hasMore, false);
  assert.deepEqual(second.rest, []);
});

test("the opening message is the earliest one, not the first row returned", () => {
  // A local Dexie read comes back in index order, so the array can start on a
  // later turn.
  const later: MessageRecord = {
    ...userMessage("a", "a later question entirely"),
    id: "a-m9",
    createdAt: 99,
  };
  const opening: MessageRecord = {
    ...userMessage("a", LONG),
    id: "a-m1",
    createdAt: 1,
  };
  const legacy = LONG.slice(0, 48) + "...";

  assert.deepEqual(
    planLegacyTitleRepairs(
      [thread("a", legacy)],
      new Map([["a", [later, opening]]]),
    ),
    [{ threadId: "a", previousTitle: legacy, title: LONG }],
  );
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



test("a chat with nothing stored is left for a later refresh", () => {
  // Its messages may simply not be imported yet. The row is not written off,
  // and once they land a later pass rewrites the title from them.
  const legacy = LONG.slice(0, 48) + "...";
  const candidates = [thread("a", legacy)];
  const messages = new Map<string, MessageRecord[]>();

  assert.deepEqual(planLegacyTitleRepairs(candidates, messages), []);
  assert.deepEqual(threadsMissingMessages(["a"], messages), ["a"]);

  messages.set("a", [userMessage("a", LONG)]);
  assert.deepEqual(threadsMissingMessages(["a"], messages), []);
  assert.deepEqual(planLegacyTitleRepairs(candidates, messages), [
    { threadId: "a", previousTitle: legacy, title: LONG },
  ]);
});

test("a rename drops the rewrite even where the guard is not enforced", () => {
  // The desktop app can meet an older backend, which ignores expectedTitle and
  // would apply the write. This check is what stops it there.
  const legacy = LONG.slice(0, 48) + "...";
  const repairs = [
    { threadId: "a", previousTitle: legacy, title: LONG },
    { threadId: "b", previousTitle: legacy, title: LONG },
    { threadId: "c", previousTitle: legacy, title: LONG },
  ];
  const current = new Map([
    ["a", legacy],
    // Renamed since the page was planned.
    ["b", "what the user typed"],
    // "c" is missing: the thread is gone.
  ]);

  assert.deepEqual(
    repairsStillValid(repairs, current).map((r) => r.threadId),
    ["a"],
  );
});
