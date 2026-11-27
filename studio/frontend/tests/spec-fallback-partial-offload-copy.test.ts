// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

// The helper lives in a .tsx this runner cannot import, so it is read as source.
const settings = readSrc("features/chat/chat-settings-sheet.tsx");

test("the Hybrid Mamba partial-offload stand-down has its own notice", () => {
  const branch = settings.match(
    /case "mtp_partial_offload":[\s\S]*?return "([^"]+)";/,
  );
  assert.ok(branch, "specFallbackMessage has no mtp_partial_offload case");

  // The default case blames the llama.cpp build; Auto hits this on builds that support MTP.
  const copy = branch[1];
  assert.doesNotMatch(copy, /llama\.cpp|update/i);
  assert.match(copy, /Settings/);

  assert.doesNotMatch(copy, /not fit|cannot fit|doesn't fit|too (big|large)/i);

  // The partial verdict includes MTP's reserve, so all layers may fit once MTP is off.
  assert.doesNotMatch(
    copy,
    /(only|just) part of this model is on the gpu|part of this model is running on/i,
  );
  assert.match(copy, /with mtp|mtp('s)? (extra state|on)|would/i);
});

test("the forced-ngram stand-down reaches the settings notice", () => {
  const gate = settings.match(/const showSpecFallback =[\s\S]*?;\n/);
  assert.ok(gate, "showSpecFallback moved");
  assert.match(gate[0], /speculativeType === "ngram"/);

  // ngram-mod opens no drafter, so the requested mode must be tested before spec_drafter_kind.
  const label = settings.match(
    /const speculativeDrafterLabel:[\s\S]*?;\n/,
  );
  assert.ok(label, "speculativeDrafterLabel moved");
  assert.match(label[0], /"ngram-mod"/);
  assert.ok(
    label[0].indexOf('loadedSpeculativeType === "ngram"') <
      label[0].indexOf("specDrafterKind"),
    "the ngram check must precede the drafter-kind checks",
  );

  assert.doesNotMatch(
    label[0],
    /(?<!loaded)[sS]peculativeType === "ngram"/,
    "the label must come from loadedSpeculativeType",
  );
});
