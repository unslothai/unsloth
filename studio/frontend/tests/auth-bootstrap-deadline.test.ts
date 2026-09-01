// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The countdown shown while an exposed first-run Studio still has its default
 * password. The server shuts itself down on that deadline, and before this the
 * browser was never told, so the session died behind a generic "Failed to load
 * auth status."
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  deadlineFromStatus,
  formatCountdown,
  hasExpired,
} from "../src/features/auth/bootstrap-deadline.ts";

test("a server that does not send the field is not time-boxed", () => {
  // A pre-deadline backend omits the key entirely. Rendering a countdown from
  // undefined would print "NaN minutes".
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
  // 0 means "expired", which must still show the banner, unlike null.
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
  // The tab keeps ticking after the deadline passes; "-12 minutes" would be worse
  // than saying nothing.
  assert.equal(formatCountdown(-1), "0 seconds");
  assert.equal(formatCountdown(-10_000_000), "0 seconds");
});

test("expiry is inclusive of zero", () => {
  assert.equal(hasExpired(1), false);
  assert.equal(hasExpired(0), true);
  assert.equal(hasExpired(-1), true);
});

test("the boundary between the two messages is one tick wide", () => {
  // At exactly 0 the wording must have already switched, so no render can show
  // "shuts down in 0 seconds".
  assert.equal(hasExpired(0), true);
  assert.equal(formatCountdown(0), "0 seconds");
  assert.ok(
    hasExpired(0),
    "0 must select the shutting-down copy, not the countdown",
  );
});
