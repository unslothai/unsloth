// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Latest feed ranks trending against release age, not by repo creation date alone.

import assert from "node:assert/strict";
import test from "node:test";

import {
  freshScore,
  rankFresh,
  releaseAgeDays,
} from "../src/features/hub/lib/fresh-ranking.ts";
import { readSrc } from "./helpers/kit.ts";

const NOW = Date.parse("2026-10-15T23:00:00Z");
const daysAgo = (days: number) => new Date(NOW - days * 86_400_000).toISOString();

// Shapes from the Hub's unsloth listing on 2026-10-15.
const museSpark = {
  name: "Muse-Spark-Open-816B-A42B-GGUF",
  createdAt: daysAgo(26.5),
  updatedAt: daysAgo(0.2),
  likes: 47,
  trendingScore: 47,
};
const nemotronBolt = {
  name: "NVIDIA-Nemotron-3.5-Bolt-74B-A9B-GGUF",
  createdAt: daysAgo(1.3),
  updatedAt: daysAgo(0.2),
  likes: 40,
  trendingScore: 40,
};
const embeddingGemma = {
  name: "embeddinggemma-2-GGUF",
  createdAt: daysAgo(9.5),
  updatedAt: daysAgo(8.7),
  likes: 314,
  trendingScore: 87,
};
const qwenOld = {
  name: "Qwen3.8-27B-GGUF",
  createdAt: daysAgo(63.6),
  updatedAt: daysAgo(56),
  likes: 5123,
  trendingScore: 120,
};
const quietNew = {
  name: "LTX-2.5-FP8",
  createdAt: daysAgo(2.9),
  updatedAt: daysAgo(2.9),
  likes: 1,
  trendingScore: 1,
};
const retouched = {
  name: "Qwen-Image-2512-FP8",
  createdAt: daysAgo(288),
  updatedAt: daysAgo(2.7),
  likes: 0,
  trendingScore: 0,
};

test("a repo kept private until launch is dated by its launch, not its creation", () => {
  assert.ok(releaseAgeDays(museSpark, NOW) < 1);
  assert.ok(Math.abs(releaseAgeDays(embeddingGemma, NOW) - 9.5) < 0.01);
  // Re-touching an old repo nobody likes does not make it new.
  assert.ok(releaseAgeDays(retouched, NOW) > 280);
});

test("trending and age are weighed together", () => {
  const ranked = rankFresh(
    [qwenOld, quietNew, retouched, embeddingGemma, nemotronBolt, museSpark],
    NOW,
  ).map((model) => model.name);
  assert.deepEqual(ranked.slice(0, 3), [museSpark.name, nemotronBolt.name, embeddingGemma.name]);
  // A two-month-old hit stays below this week's models, and an untouched old repo ranks last.
  assert.ok(ranked.indexOf(qwenOld.name) > ranked.indexOf(embeddingGemma.name));
  assert.equal(ranked.at(-1), retouched.name);
  assert.ok(freshScore(quietNew, NOW) > 0);
});

test("the Latest feed uses the fresh ranking, which no Hub request ever sends", () => {
  assert.match(readSrc("features/hub/lib/channels.ts"), /id: "unsloth-latest",[\s\S]*?sort: "fresh",/);
  const hook = readSrc("features/hub/hooks/use-hub-model-search.ts");
  assert.match(hook, /if \(opts\.sortBy === "fresh"\) \{\n\s*return freshChannelIterator\(channel, opts\);/);
  assert.match(hook, /url\.searchParams\.set\("sort", hubListingSort\(sortBy\)\)/);
  assert.doesNotMatch(hook, /sort: (opts\.)?sortBy,/);
});

test("Latest picked from the sort menu, outside any section, ranks the same way", () => {
  const hook = readSrc("features/hub/hooks/use-hub-model-search.ts");
  assert.match(
    hook,
    /if \(sortBy === "fresh"\) \{[\s\S]*?freshChannelIterator\(\n\s*\{ owner: unslothOnly \? "unsloth" : undefined \},/,
  );
});
