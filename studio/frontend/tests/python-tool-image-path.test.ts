// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { pythonToolImagePath } from "../src/components/assistant-ui/python-tool-image-path.ts";

test("escapes Python tool image path segments", () => {
  assert.equal(
    pythonToolImagePath("session/id", "loss curve #1.png"),
    "/api/inference/sandbox/session%2Fid/loss%20curve%20%231.png",
  );
});

test("keeps authentication out of the Python tool image URL", () => {
  const path = pythonToolImagePath("session", "plot.png");

  assert.equal(path, "/api/inference/sandbox/session/plot.png");
  assert.ok(!path.includes("token"));
  assert.ok(!path.startsWith("http"));
});
