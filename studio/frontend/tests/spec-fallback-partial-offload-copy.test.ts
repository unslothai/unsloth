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

  // Nor may it assert the placement the model ENDS UP with. The partial verdict
  // is priced with MTP's rollback reserve included, so on the --fit path
  // llama.cpp can place every layer on the GPU once this branch turns MTP off;
  // an unconditional "only part of this model is on the GPU" would then be false
  // and would recommend the placement the load already has. Describe what MTP
  // would require instead.
  assert.doesNotMatch(
    copy,
    /(only|just) part of this model is on the gpu|part of this model is running on/i,
  );
  assert.match(copy, /with mtp|mtp('s)? (extra state|on)|would/i);
});
