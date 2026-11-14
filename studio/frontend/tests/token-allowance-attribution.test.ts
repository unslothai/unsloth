// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const ADAPTER = readSrc("features/chat/api/chat-adapter.ts");

test("live, final and failed replies all carry the allowance estimates", () => {
  const helper = ADAPTER.slice(ADAPTER.indexOf("const tokenUsageMetadata = () => ({"));
  for (const field of [
    "tokenUsageEstimatedPrompt,",
    "tokenUsageEstimatedToolOutput: tokenUsageEstimatedToolOutput(),",
    // timing adds the continued reply's count, which must not be charged twice.
    "tokenUsagePriorCount: priorTiming?.tokenCount,",
  ]) {
    assert.ok(helper.slice(0, 400).includes(field), field);
  }
  assert.equal(ADAPTER.match(/\.\.\.tokenUsageMetadata\(\),/g)?.length, 3);
});

test("the allowance switch saves the visible draft and can always turn it off", () => {
  const SECTION = readSrc("features/settings/components/token-limit-section.tsx");
  assert.match(SECTION, /onCheckedChange=\{\(next\) => void save\(\{ enabled: next \}\)\}/);
  assert.match(SECTION, /disabled=\{busy \|\| \(!enabled && cap === null\)\}/);
  assert.match(SECTION, /\{\s*period,\s*\.\.\.\(cap === null \? \{\} : \{ limit: cap \}\)/);
});

test("saving a temporary chat marks its replies as already charged", () => {
  const RUNTIME = readSrc("features/chat/runtime-provider.tsx");
  const save = RUNTIME.slice(RUNTIME.indexOf("export async function persistTemporaryThread("));
  assert.match(
    save.slice(0, 3000),
    /message\.role === "assistant" \? \{ \.\.\.metadata, tokenUsageCharged: true \} : metadata/,
  );
});

test("title requests are marked stateless so their receipt is charged", () => {
  const TITLE = readSrc("features/chat/utils/chat-title.ts");
  const build = TITLE.slice(TITLE.indexOf("export async function buildTitleRequest("));
  assert.match(build.slice(0, 2500), /temporary_chat: true,/);
});
