// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { transportAfterStart } from "../src/features/hub/download-manager/constants.ts";

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
