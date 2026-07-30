// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { ownershipFromStartResult } from "../src/features/hub/download-manager/start-ownership.ts";

test("grants ownership only when the backend confirms this caller created the job", () => {
  assert.equal(ownershipFromStartResult({ created: true }), "started");
  assert.equal(ownershipFromStartResult({ created: false }), "existing");
  assert.equal(ownershipFromStartResult({}), "existing");
});
