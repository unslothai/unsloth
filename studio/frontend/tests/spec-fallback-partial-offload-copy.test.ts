// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

// specFallbackMessage is a module-local helper inside a .tsx, which this runner
// cannot import (it strips types but does not transform JSX), so the file is read
// as source -- the same guard the other chat-settings-sheet tests use.
const settings = readFileSync(
  new URL("../src/features/chat/chat-settings-sheet.tsx", import.meta.url),
  "utf8",
);

test("the Hybrid Mamba partial-offload stand-down has its own notice", () => {
  const branch = settings.match(
    /case "mtp_partial_offload":[\s\S]*?return "([^"]+)";/,
  );
  assert.ok(branch, "specFallbackMessage has no mtp_partial_offload case");

  // Without a case it falls through to the default, which blames the installed
  // llama.cpp build and offers an update. Auto took this path on a build that
  // does support MTP, so both halves would be wrong; the remedy is to force it.
  const copy = branch[1];
  assert.doesNotMatch(copy, /llama\.cpp|update/i);
  assert.match(copy, /Settings/);

  // And it must not diagnose a failed fit. Manual mode reaches this with a
  // partial layer count the user picked, on a card that may hold the whole
  // model, where the useful remedy is more layers rather than forcing MTP.
  assert.doesNotMatch(copy, /not fit|cannot fit|doesn't fit|too (big|large)/i);
});
