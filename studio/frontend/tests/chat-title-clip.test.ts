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
  threadsAwaitingImport,
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
  // 120 UTF-16 units including the ellipsis, which is what the input accepts.
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
  // A lone surrogate survives JSON.stringify but 500s the backend's SQLite bind.
  const line = "x".repeat(119) + "\u{1F600} tail";
  // A raw cut at the budget lands mid-pair.
  assert.equal(UNPAIRED_SURROGATE.test(line.slice(0, 120)), true);
  const title = fallbackTitleFromUserText(line);
  assert.equal(UNPAIRED_SURROGATE.test(title), false);
  // The emoji needs two units and only one is left, so it is left out whole.
  assert.equal(title, "x".repeat(119) + "…");
  assert.equal(title.length, 120);
});

test("a lone surrogate is dropped even when the line is under the cap", () => {
  // The cut sanitises what it walks, so under-cap lines used to be stored as
  // they came, and one unpaired surrogate 500s the backend's title write.
  const line = "x".repeat(60) + "\uD83D";
  assert.ok(line.length <= 120);
  assert.equal(UNPAIRED_SURROGATE.test(line), true);
  const title = fallbackTitleFromUserText(line);
  assert.equal(UNPAIRED_SURROGATE.test(title), false);
  assert.equal(title, "x".repeat(60));
  // A trailing low surrogate with no high before it goes the same way.
  assert.equal(fallbackTitleFromUserText("hi \uDE00"), "hi");
  // A valid pair under the cap is untouched.
  assert.equal(fallbackTitleFromUserText("hi \u{1F600}"), "hi \u{1F600}");
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
    { threadId: "a", previousTitle: legacy, openingMessageId: "a-m1", title: LONG },
  ]);
});

test("a drain advances even when a whole page failed and was unmarked", () => {
  // Failures get unmarked for a later refresh. Selecting the next page off the
  // same list would draw them straight back in and never reach the rest.
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
  // A local read comes back in index order, so it can start on a later turn.
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
    // Guarded on the opening message, not the row the array happens to start on.
    [{ threadId: "a", previousTitle: legacy, openingMessageId: "a-m1", title: LONG }],
  );
});

test("two prompts sharing a timestamp break on id, as the backend does", () => {
  // The write is guarded on this id, so both orders must pick the same message.
  const legacy = LONG.slice(0, 48) + "...";
  const first: MessageRecord = { ...userMessage("a", LONG), id: "a-m1" };
  const second: MessageRecord = {
    ...userMessage("a", "a different question"),
    id: "a-m2",
  };

  for (const order of [
    [first, second],
    [second, first],
  ]) {
    assert.deepEqual(
      planLegacyTitleRepairs([thread("a", legacy)], new Map([["a", order]])),
      [
        {
          threadId: "a",
          previousTitle: legacy,
          openingMessageId: "a-m1",
          title: LONG,
        },
      ],
    );
  }
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
  // A not-yet-imported chat reads empty; an unknown id is missing from the map.
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
  // Its messages may not be imported yet, so a later pass rewrites the title.
  const legacy = LONG.slice(0, 48) + "...";
  const candidates = [thread("a", legacy)];
  const messages = new Map<string, MessageRecord[]>();

  assert.deepEqual(planLegacyTitleRepairs(candidates, messages), []);
  assert.deepEqual(threadsMissingMessages(["a"], messages), ["a"]);

  messages.set("a", [userMessage("a", LONG)]);
  assert.deepEqual(threadsMissingMessages(["a"], messages), []);
  assert.deepEqual(planLegacyTitleRepairs(candidates, messages), [
    { threadId: "a", previousTitle: legacy, openingMessageId: "a-m1", title: LONG },
  ]);
});

test("a chat whose opening prompt is gone is decided, not retried forever", () => {
  // A chat that does have messages is a complete answer: the opening prompt was
  // deleted or edited, so no later pass can prove the title. Unmarking it would
  // re-select it on every refresh, since its title stays clipped.
  const legacy = LONG.slice(0, 48) + "...";
  const candidates = [thread("a", legacy)];
  const messages = new Map<string, MessageRecord[]>([
    ["a", [{ ...userMessage("a", "a different question entirely"), id: "a-m9" }]],
  ]);

  assert.deepEqual(planLegacyTitleRepairs(candidates, messages), []);
  assert.deepEqual(threadsMissingMessages(["a"], messages), []);
  // So it stays marked, and the next page passes over it.
  assert.deepEqual(
    selectLegacyRepairPage(candidates, new Set(["a"]), 100).candidates,
    [],
  );
});

test("an emptied chat is decided, one still importing is not", () => {
  // Both read back as zero messages. The ledger tells them apart: one it knows
  // was imported is simply empty, one it has never seen may still be on its way.
  const ids = ["emptied", "importing", "fine"];
  const messages = new Map<string, MessageRecord[]>([
    ["emptied", []],
    ["importing", []],
    ["fine", [userMessage("fine", LONG)]],
  ]);

  assert.deepEqual(threadsMissingMessages(ids, messages), [
    "emptied",
    "importing",
  ]);
  assert.deepEqual(
    threadsAwaitingImport(ids, messages, new Set(["emptied"])),
    ["importing"],
  );
  // An unreadable ledger decides nothing, so both stay retryable.
  assert.deepEqual(threadsAwaitingImport(ids, messages, new Set()), [
    "emptied",
    "importing",
  ]);
});
