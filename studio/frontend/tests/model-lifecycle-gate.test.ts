import assert from "node:assert/strict";
import test from "node:test";

import { ModelLifecycleGate } from "../src/features/chat/utils/model-lifecycle-gate.ts";

test("the model lifecycle has one owner and ignores stale releases", () => {
  const gate = new ModelLifecycleGate();
  const first = gate.tryAcquire();

  assert.notEqual(first, null);
  if (first === null) {
    assert.fail("expected the first lifecycle lease");
  }
  assert.equal(gate.tryAcquire(), null);
  assert.equal(gate.release(first + 1), false);
  assert.equal(gate.tryAcquire(), null);
  assert.equal(gate.release(first), true);

  const second = gate.tryAcquire();
  assert.notEqual(second, null);
  assert.notEqual(second, first);
});
