// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  estimatePromptTokens,
  lastMeasuredPromptRate,
  promptProgressMetrics,
} from "../src/features/chat/utils/generation-progress.ts";

test("prompt size includes messages and tool definitions", () => {
  const messagesOnly = estimatePromptTokens({
    messages: [{ role: "user", content: "Explain the result" }],
  });
  const withTool = estimatePromptTokens({
    messages: [{ role: "user", content: "Explain the result" }],
    tools: [
      {
        type: "function",
        function: {
          name: "lookup_result",
          description: "Look up a stored experiment result by identifier",
        },
      },
    ],
  });

  assert.ok(messagesOnly !== undefined);
  assert.ok(withTool !== undefined);
  assert.ok(withTool > messagesOnly);
});

test("prompt size does not treat encoded media as text tokens", () => {
  const estimated = estimatePromptTokens({
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: "Describe this image" },
          Object.fromEntries([
            ["type", "image_url"],
            ["image_url", `data:image/png;base64,${"a".repeat(50_000)}`],
          ]),
        ],
      },
    ],
  });

  assert.ok(estimated !== undefined);
  assert.ok(estimated < 100);
});

test("the latest valid measured prompt rate is used as a baseline", () => {
  const rate = lastMeasuredPromptRate([
    {
      metadata: {
        custom: {
          serverTimings: Object.fromEntries([
            ["prompt_n", 1_000],
            ["prompt_ms", 500],
            ["prompt_per_second", 2_000],
          ]),
        },
      },
    },
    {
      metadata: {
        custom: {
          serverTimings: Object.fromEntries([
            ["prompt_n", 10],
            ["prompt_ms", 0],
            ["prompt_per_second", Number.POSITIVE_INFINITY],
          ]),
        },
      },
    },
  ]);

  assert.equal(rate, 2_000);
});

test("real prompt progress produces percentage, uncached throughput, and ETA", () => {
  const metrics = promptProgressMetrics({
    total: 1_000,
    processed: 600,
    cache: 100,
    timeMs: 250,
  });

  assert.equal(metrics.percentage, 60);
  assert.equal(metrics.tokensPerSecond, 2_000);
  assert.equal(metrics.etaMs, 200);
});
