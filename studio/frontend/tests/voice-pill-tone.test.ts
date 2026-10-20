// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";
import { voicePillIconClass } from "../src/features/chat/voice/voice-pill-tone.ts";

test("the Voice pill icon turns green only when every role is loaded", () => {
  const green = "text-emerald-500";
  assert.equal(
    voicePillIconClass(["loaded", "loaded", "loaded", "loaded", "loaded"]),
    green,
  );
  // A role with no state (Decide for a non-owner) does not hold it back.
  assert.equal(
    voicePillIconClass(["loaded", "loaded", null, "loaded", "loaded"]),
    green,
  );
  for (const other of ["not_loaded", "loading", "error"]) {
    assert.equal(
      voicePillIconClass(["loaded", other, "loaded", "loaded", "loaded"]),
      "text-muted-foreground",
    );
  }
  assert.equal(voicePillIconClass([null, null]), "text-muted-foreground");
});
