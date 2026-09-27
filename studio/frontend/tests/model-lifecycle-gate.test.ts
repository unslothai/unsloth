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

test("queue acceptance follows the owning lifecycle phase", () => {
  for (const phase of ["preparing", "loading", "unloading"] as const) {
    const gate = new ModelLifecycleGate();
    assert.equal(gate.canQueue(), true);
    const lease = gate.tryAcquire(phase)!;
    assert.equal(gate.canQueue(), phase === "loading");
    assert.equal(gate.tryAcquire("loading"), null);
    assert.equal(gate.markLoading(lease + 1), false);
    assert.equal(gate.canQueue(), phase === "loading");
    assert.equal(gate.markLoading(lease), phase === "preparing");
    assert.equal(gate.canQueue(), phase !== "unloading");
    assert.equal(gate.release(lease), true);
    assert.equal(gate.canQueue(), true);
    const next = gate.tryAcquire("preparing")!;
    assert.equal(gate.markLoading(lease), false);
    assert.equal(gate.release(lease), false);
    assert.equal(gate.canQueue(), false);
    assert.equal(gate.release(next), true);
  }
});

test("legacy lifecycle callers keep drafts blocked until release or an explicit loading boundary", () => {
  const gate = new ModelLifecycleGate();
  const lease = gate.tryAcquire()!;
  assert.equal(gate.canQueue(), false);
  assert.equal(gate.markLoading(lease + 1), false);
  assert.equal(gate.canQueue(), false);
  assert.equal(gate.markLoading(lease), true);
  assert.equal(gate.canQueue(), true);
  gate.release(lease);

  const next = gate.tryAcquire()!;
  assert.equal(gate.canQueue(), false);
  assert.equal(gate.markLoading(lease), false);
  assert.equal(gate.canQueue(), false);
  gate.release(next);
  assert.equal(gate.canQueue(), true);
});

test("failure blocks new drafts throughout rollback and rejects stale callbacks", () => {
  const gate = new ModelLifecycleGate();
  const lease = gate.tryAcquire("loading")!;
  assert.equal(gate.markFailed(lease + 1), false);
  assert.equal(gate.canQueue(), true);
  assert.equal(gate.markFailed(lease), true);
  assert.equal(gate.canQueue(), false);
  assert.equal(gate.markFailed(lease), false);
  gate.release(lease);
  const next = gate.tryAcquire("loading")!;
  assert.equal(gate.markFailed(lease), false);
  assert.equal(gate.canQueue(), true);
  gate.release(next);
});
