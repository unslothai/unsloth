// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A re-pointed run keeps its id, so the old reply would otherwise render the new card twice.

import assert from "node:assert/strict";
import test from "node:test";

import { researchReplyOwnsRun } from "../src/features/chat/utils/research-run-binding.ts";

test("the reply the run points at owns it", () => {
  assert.equal(researchReplyOwnsRun("assistant-1", "assistant-1"), true);
});

test("the reply the run was re-pointed away from does not", () => {
  assert.equal(researchReplyOwnsRun("assistant-2", "assistant-1"), false);
});

test("a run that named no reply belongs to whoever asks", () => {
  for (const bound of [undefined, null, ""]) {
    assert.equal(
      researchReplyOwnsRun(bound, "assistant-1"),
      true,
      String(bound),
    );
  }
});

test("a message with no id of its own is left as it was", () => {
  for (const id of [undefined, null, ""]) {
    assert.equal(researchReplyOwnsRun("assistant-2", id), true, String(id));
  }
});
