// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const ADAPTER = readSrc("features/chat/api/chat-adapter.ts");

test("only external requests stamp a provider for the token allowance", () => {
  assert.match(
    ADAPTER,
    /const tokenUsageProvider = isExternalRequest\s*\?\s*externalProvider\?\.providerType\s*:\s*undefined;/,
  );
});

test("live, final and failed replies all carry the provider stamp", () => {
  const live = ADAPTER.slice(ADAPTER.indexOf("const liveCustom = () => ({"));
  assert.match(live.slice(0, 300), /tokenUsageProvider,/);
  const failed = ADAPTER.slice(
    ADAPTER.indexOf("const partialTiming = withPriorTiming("),
    ADAPTER.indexOf("resumeTiming: passSoFar ? addPass(priorTiming, passSoFar()) : priorTiming,\n                ...generationCustom()"),
  );
  assert.match(failed, /tokenUsageProvider,/);
  const final = ADAPTER.slice(
    ADAPTER.lastIndexOf("timing: finalTiming,\n            custom: {"),
    ADAPTER.indexOf("responseDetails: buildResponseDetails(finishedAt),"),
  );
  assert.match(final, /tokenUsageProvider,/);
});
