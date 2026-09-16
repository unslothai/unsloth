// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const featureRoot = new URL("../src/features/profile/", import.meta.url);

test("profile stats type exposes combined chat and API token breakdowns", async () => {
  const source = await readFile(
    new URL("api/profile-stats.ts", featureRoot),
    "utf8",
  );
  for (const field of [
    "chatPromptTokens",
    "chatCompletionTokens",
    "chatTokens",
    "apiPromptTokens",
    "apiCompletionTokens",
    "apiTokens",
  ]) {
    assert.match(source, new RegExp(`${field}: number`));
  }
});

test("API-only users render activity and the token breakdown stays semantically split", async () => {
  const [content, insights] = await Promise.all([
    readFile(
      new URL("components/stats/profile-stats-content.tsx", featureRoot),
      "utf8",
    ),
    readFile(
      new URL("components/stats/insights-card.tsx", featureRoot),
      "utf8",
    ),
  ]);

  assert.match(
    content,
    /stats\.totals\.messages > 0 \|\| stats\.totals\.totalTokens > 0/,
  );
  assert.match(insights, /totals\.chatTokens \/ totals\.threads/);
  assert.match(insights, /totals\.cachedTokens \/ totals\.chatPromptTokens/);
  assert.match(insights, /settings\.profile\.stats\.totalTokens/);
  assert.match(insights, /settings\.profile\.stats\.studioChatTokens/);
  assert.match(insights, /settings\.profile\.stats\.apiTokens/);
});
