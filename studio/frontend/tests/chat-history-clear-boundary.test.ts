import assert from "node:assert/strict";
import test from "node:test";

import { ChatHistoryClearBoundary } from "../src/features/chat/utils/chat-history-clear-boundary.ts";

test("a history clear invalidates work captured before it", () => {
  const boundary = new ChatHistoryClearBoundary();
  const captured = boundary.capture();
  assert.equal(boundary.capture(), captured);
  boundary.advance();
  assert.notEqual(boundary.capture(), captured);
});

test("a history clear can drain persistence accepted before it", async () => {
  const boundary = new ChatHistoryClearBoundary();
  let finish!: () => void;
  const pending = new Promise<void>((resolve) => {
    finish = resolve;
  });
  boundary.trackPending(pending);
  let drained = false;
  const wait = boundary.waitForPending().then(() => {
    drained = true;
  });
  await Promise.resolve();
  assert.equal(drained, false);
  finish();
  await wait;
  assert.equal(drained, true);
});
