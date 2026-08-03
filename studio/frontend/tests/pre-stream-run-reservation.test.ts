import assert from "node:assert/strict";
import test from "node:test";

import {
  adoptPreStreamRunReservation,
  cancelPreStreamRunReservations,
  findPreStreamRunReservation,
  hasPreStreamRunReservation,
  isPreStreamRunReservationCancelled,
  listLocalPreStreamRunReservations,
  preStreamRunThreadIdsForAdapter,
  preStreamRunThreadIdsForRuntime,
  releasePreStreamRunForThreadIds,
  releasePreStreamRunReservation,
  reservePreStreamRun,
} from "../src/features/chat/utils/pre-stream-run-reservation.ts";

test("adapter thread ids never mix an identified background run with the visible chat", () => {
  assert.deepEqual(
    preStreamRunThreadIdsForAdapter("background-thread", "visible-thread"),
    ["background-thread"],
  );
  assert.deepEqual(
    preStreamRunThreadIdsForAdapter(undefined, "visible-thread"),
    ["visible-thread"],
  );
  assert.deepEqual(preStreamRunThreadIdsForAdapter(undefined, null), []);
});

test("runtime thread ids use the visible chat only when no runtime identity exists", () => {
  assert.deepEqual(
    preStreamRunThreadIdsForRuntime(
      ["remote-thread", "local-thread"],
      "visible-thread",
    ),
    ["remote-thread", "local-thread"],
  );
  assert.deepEqual(
    preStreamRunThreadIdsForRuntime([null, undefined], "visible-thread"),
    ["visible-thread"],
  );
  assert.deepEqual(
    preStreamRunThreadIdsForRuntime(["thread", "thread"], null),
    ["thread"],
  );
});

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

test("local reservations can be snapshotted and cancelled before streaming", () => {
  let cancelCount = 0;
  const local = reservePreStreamRun(["local-thread"], {
    usesLocalModel: true,
    cancel: (threadIds) => {
      assert.deepEqual(threadIds, ["local-thread", "remote-thread"]);
      cancelCount += 1;
    },
  });
  const external = reservePreStreamRun(["external-thread"], {
    usesLocalModel: false,
  });
  assert.ok(local);
  assert.ok(external);
  assert.equal(
    adoptPreStreamRunReservation(local, ["remote-thread"]),
    true,
  );
  assert.deepEqual(listLocalPreStreamRunReservations(), [
    { token: local, threadIds: ["local-thread", "remote-thread"] },
  ]);
  assert.equal(cancelPreStreamRunReservations([local]), 1);
  assert.equal(cancelCount, 1);
  assert.equal(isPreStreamRunReservationCancelled(local), true);
  assert.deepEqual(listLocalPreStreamRunReservations(), []);
  assert.equal(cancelPreStreamRunReservations([local]), 0);
  assert.equal(releasePreStreamRunReservation(local), true);
  assert.equal(releasePreStreamRunReservation(external), true);
});
