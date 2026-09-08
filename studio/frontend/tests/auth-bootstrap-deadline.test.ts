// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  deadlineFromStatus,
  formatCountdown,
  hasExpired,
} from "../src/features/auth/bootstrap-deadline.ts";

test("a server that does not send the field is not time-boxed", () => {
  // A pre-deadline backend omits the key; a countdown from undefined prints "NaN".
  assert.equal(deadlineFromStatus(undefined, 1_000_000), null);
  assert.equal(deadlineFromStatus(null, 1_000_000), null);
});

test("a loopback launch sends null and gets no countdown", () => {
  assert.equal(deadlineFromStatus(null, 0), null);
});

test("seconds are turned into an absolute expiry", () => {
  // Absolute, so a backgrounded tab whose timers stopped still renders correctly.
  assert.equal(deadlineFromStatus(3600, 1_000_000), 1_000_000 + 3_600_000);
});

test("zero seconds is a real deadline, not an absent one", () => {
  assert.equal(deadlineFromStatus(0, 500), 500);
});

test("a non-numeric or non-finite value is refused", () => {
  assert.equal(deadlineFromStatus(Number.NaN, 0), null);
  assert.equal(deadlineFromStatus(Number.POSITIVE_INFINITY, 0), null);
  assert.equal(deadlineFromStatus("3600" as unknown as number, 0), null);
});

test("under a minute reads in seconds", () => {
  assert.equal(formatCountdown(45_000), "45 seconds");
  assert.equal(formatCountdown(59_400), "59 seconds");
});

test("one second is singular", () => {
  assert.equal(formatCountdown(1000), "1 second");
});

test("a minute or more reads in minutes", () => {
  assert.equal(formatCountdown(60_000), "1 minute");
  assert.equal(formatCountdown(3_600_000), "60 minutes");
  assert.equal(formatCountdown(2_520_000), "42 minutes");
});

test("it never renders a negative", () => {
  // The tab keeps ticking past the deadline; "-12 minutes" is worse than nothing.
  assert.equal(formatCountdown(-1), "0 seconds");
  assert.equal(formatCountdown(-10_000_000), "0 seconds");
});

test("expiry is inclusive of zero", () => {
  assert.equal(hasExpired(1), false);
  assert.equal(hasExpired(0), true);
  assert.equal(hasExpired(-1), true);
});

test("the boundary between the two messages is one tick wide", () => {
  assert.equal(hasExpired(0), true);
  assert.equal(formatCountdown(0), "0 seconds");
  assert.ok(
    hasExpired(0),
    "0 must select the shutting-down copy, not the countdown",
  );
});

test("the deadline and the clock it is compared against must be one sample", () => {
  // A request that takes 30ms turns a server 0 back into a positive remainder,
  // which selects the countdown copy and prints "shuts down in 0 seconds".
  const mounted = 1_000_000;
  const responded = mounted + 30;
  const stale = deadlineFromStatus(0, responded);
  assert.ok(stale !== null);
  assert.equal(hasExpired(stale - mounted), false);
  assert.equal(formatCountdown(stale - mounted), "0 seconds");

  const sampled = deadlineFromStatus(0, responded);
  assert.ok(sampled !== null);
  assert.equal(hasExpired(sampled - responded), true);
});
