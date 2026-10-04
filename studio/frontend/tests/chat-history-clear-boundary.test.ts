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
