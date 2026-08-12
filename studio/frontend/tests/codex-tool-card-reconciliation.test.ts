// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { resolveToolCallPartId } from "../src/features/chat/tool-call-id.ts";

test("a Codex delta and execution events resolve to one card id", () => {
  const ids = new Map<string, string>();
  let sequence = 0;
  const createId = () => `call-1:run-${++sequence}`;

  const deltaId = resolveToolCallPartId(ids, "call-1", undefined, "", createId);
  const startId = resolveToolCallPartId(ids, "call-1", undefined, "", createId);
  const endId = resolveToolCallPartId(ids, "call-1", undefined, "", createId);

  assert.equal(deltaId, startId);
  assert.equal(startId, endId);
  assert.equal(sequence, 1);
});

test("confirmation ids and empty backend ids keep their existing identity", () => {
  const ids = new Map<string, string>();
  assert.equal(
    resolveToolCallPartId(ids, "call-1", "approval-1", "", () => "new"),
    "approval-1",
  );
  assert.equal(
    resolveToolCallPartId(ids, "", undefined, "last-card", () => "new"),
    "last-card",
  );
});
