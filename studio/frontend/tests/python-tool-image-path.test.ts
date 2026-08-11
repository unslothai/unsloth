// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { pythonToolImagePath } from "../src/components/assistant-ui/python-tool-image-path.ts";

test("escapes Python tool image path segments", () => {
  // An id with a path separator now travels in the query rather than in the
  // path: an encoded slash is rejected or decoded by proxies before the route
  // sees it, so it did not survive the round trip.
  assert.equal(
    pythonToolImagePath("session/id", "loss curve #1.png"),
    "/api/inference/sandbox/_/loss%20curve%20%231.png?session=session%2Fid",
  );
});

test("keeps authentication out of the Python tool image URL", () => {
  const path = pythonToolImagePath("session", "plot.png");

  assert.equal(path, "/api/inference/sandbox/session/plot.png");
  assert.ok(!path.includes("token"));
  assert.ok(!path.startsWith("http"));
});
