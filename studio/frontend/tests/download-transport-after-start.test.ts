// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  adoptedTransports,
  probeDescribesCurrentRun,
  transportAfterStart,
} from "../src/features/hub/download-manager/constants.ts";

test("a start of its own keeps the transport it resolved", () => {
  assert.equal(transportAfterStart("http", null), "http");
  assert.equal(transportAfterStart("xet", undefined), "xet");
});

test("adopting another client's run takes that run's transport", () => {
  // Requested HTTP, attached to a Xet job: Pause would promise a resume the
  // Xet worker cannot give.
  assert.equal(transportAfterStart("http", "xet"), "xet");
  assert.equal(transportAfterStart("xet", "http"), "http");
});

test("an unresolved or unknown report is ignored, not trusted", () => {
  assert.equal(transportAfterStart("http", "auto"), "http");
  assert.equal(transportAfterStart("xet", "ftp"), "xet");
  assert.equal(transportAfterStart("xet", 3), "xet");
});

test("a probe for the run a job is on may patch it", () => {
  assert.equal(probeDescribesCurrentRun(7, 7), true);
});

test("a probe from before a cancel and restart may not", () => {
  // The old response would otherwise hand the restarted job the old
  // transport, and the fresh start's own reply cannot repair it.
  assert.equal(probeDescribesCurrentRun(8, 7), false);
});

test("a job with no generation yet takes what the probe reports", () => {
  assert.equal(probeDescribesCurrentRun(undefined, 7), true);
  assert.equal(probeDescribesCurrentRun(null, 0), true);
});

test("a probe with no generation proves nothing", () => {
  assert.equal(probeDescribesCurrentRun(undefined, undefined), false);
  assert.equal(probeDescribesCurrentRun(7, undefined), false);
  assert.equal(probeDescribesCurrentRun(7, 7.5), false);
});

test("adoption keeps a persisted marker when the probe carries none", () => {
  // `/download-status` has no cancel marker and can win the hydration race
  // against `/active-downloads`, which does.
  assert.deepEqual(
    adoptedTransports(
      { transport: "http" },
      { transport: "http", cancelTransport: "xet" },
    ),
    { transport: "http", cancelTransport: "xet" },
  );
});

test("a probe that reports both wins over what was stored", () => {
  assert.deepEqual(
    adoptedTransports(
      { transport: "xet", cancelTransport: "http" },
      { transport: "http", cancelTransport: "xet" },
    ),
    { transport: "xet", cancelTransport: "http" },
  );
});

test("an adoption with nothing stored and nothing reported holds neither", () => {
  assert.deepEqual(adoptedTransports({}, undefined), {
    transport: undefined,
    cancelTransport: undefined,
  });
});

test("the persisted transport survives a probe that omits it too", () => {
  assert.deepEqual(
    adoptedTransports({ cancelTransport: "xet" }, { transport: "http" }),
    { transport: "http", cancelTransport: "xet" },
  );
});

test("a reported null clears a marker left by an earlier run", () => {
  // /active-downloads always carries the field, so null is the backend saying
  // this run has no marker, not a source that cannot report one.
  assert.deepEqual(
    adoptedTransports(
      { transport: "http", cancelTransport: null },
      { transport: "xet", cancelTransport: "xet" },
    ),
    { transport: "http", cancelTransport: undefined },
  );
});

test("a reported null with nothing stored still holds no marker", () => {
  assert.deepEqual(
    adoptedTransports({ transport: "http", cancelTransport: null }, undefined),
    { transport: "http", cancelTransport: undefined },
  );
});
