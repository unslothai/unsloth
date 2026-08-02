import assert from "node:assert/strict";
import test from "node:test";

import {
  adoptPreStreamRunReservation,
  findPreStreamRunReservation,
  hasPreStreamRunReservation,
  releasePreStreamRunForThreadIds,
  releasePreStreamRunReservation,
  reservePreStreamRun,
} from "../src/features/chat/utils/pre-stream-run-reservation.ts";

test("pre-stream reservations are immediate and scoped per thread", () => {
  const first = reservePreStreamRun(["thread-a"]);
  assert.ok(first);
  assert.equal(hasPreStreamRunReservation(["thread-a"]), true);
  assert.equal(reservePreStreamRun(["thread-a"]), null);

  const sibling = reservePreStreamRun(["thread-b"]);
  assert.ok(sibling);
  assert.notEqual(sibling, first);

  assert.equal(releasePreStreamRunReservation(first), true);
  assert.equal(hasPreStreamRunReservation(["thread-a"]), false);
  assert.equal(hasPreStreamRunReservation(["thread-b"]), true);
  assert.equal(releasePreStreamRunReservation(sibling), true);
});

test("a reservation follows thread hydration and releases every alias", () => {
  const token = reservePreStreamRun(["local-thread"]);
  assert.ok(token);
  assert.equal(
    adoptPreStreamRunReservation(token, ["local-thread", "remote-thread"]),
    true,
  );
  assert.equal(findPreStreamRunReservation(["remote-thread"]), token);
  assert.equal(releasePreStreamRunForThreadIds(["remote-thread"]), true);
  assert.equal(hasPreStreamRunReservation(["local-thread"]), false);
  assert.equal(hasPreStreamRunReservation(["remote-thread"]), false);
  assert.equal(releasePreStreamRunReservation(token), false);
});

test("alias adoption cannot steal another thread reservation", () => {
  const first = reservePreStreamRun(["first-local"]);
  const second = reservePreStreamRun(["second-local", "shared-remote"]);
  assert.ok(first);
  assert.ok(second);
  assert.equal(
    adoptPreStreamRunReservation(first, ["first-local", "shared-remote"]),
    false,
  );
  assert.equal(findPreStreamRunReservation(["shared-remote"]), second);
  assert.equal(releasePreStreamRunReservation(first), true);
  assert.equal(releasePreStreamRunReservation(second), true);
});
