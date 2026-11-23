// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Latest feed ranks trending against creation date, never against last modified.

import assert from "node:assert/strict";
import test from "node:test";

import {
  createdAgeDays,
  freshScore,
  rankFresh,
} from "../src/features/hub/lib/fresh-ranking.ts";
import { readSrc } from "./helpers/kit.ts";

const NOW = Date.parse("2026-10-20T10:00:00Z");
const daysAgo = (days: number) =>
  new Date(NOW - days * 86_400_000).toISOString();

// Shapes from the Hub's unsloth listing on 2026-10-20.
const deepseekFlash = {
  name: "DeepSeek-V4.1-Flash-1017-GGUF",
  createdAt: daysAgo(2.6),
  updatedAt: daysAgo(0.4),
  trendingScore: 85,
};
const nemotronBolt = {
  name: "NVIDIA-Nemotron-3.5-Bolt-74B-A9B-GGUF",
  createdAt: daysAgo(5.8),
  updatedAt: daysAgo(3.9),
  trendingScore: 76,
};
// Re-uploaded 3 hours ago; created a week ago.
const lfm = {
  name: "LFM2.5-24B-A2B-GGUF",
  createdAt: daysAgo(6.8),
  updatedAt: daysAgo(0.1),
  trendingScore: 89,
};
const embeddingGemma = {
  name: "embeddinggemma-2-GGUF",
  createdAt: daysAgo(14),
  updatedAt: daysAgo(13),
  trendingScore: 43,
};
const quietNew = {
  name: "sam3.1",
  createdAt: daysAgo(0.3),
  updatedAt: daysAgo(0.3),
  trendingScore: 2,
};
const retouched = {
  name: "Qwen-Image-2512-FP8",
  createdAt: daysAgo(293),
  updatedAt: daysAgo(0.5),
  trendingScore: 0,
};

test("age is counted from creation, so a re-upload is not new", () => {
  assert.ok(Math.abs(createdAgeDays(lfm, NOW) - 6.8) < 0.01);
  assert.ok(createdAgeDays(retouched, NOW) > 290);
  assert.equal(
    createdAgeDays({ trendingScore: 5 }, NOW),
    Number.POSITIVE_INFINITY,
  );
});

test("trending and creation date are weighed together", () => {
  const ranked = rankFresh(
    [retouched, quietNew, embeddingGemma, lfm, nemotronBolt, deepseekFlash],
    NOW,
  ).map((model) => model.name);
  // The re-uploaded LFM no longer jumps above models released after it.
  assert.deepEqual(ranked.slice(0, 3), [
    deepseekFlash.name,
    nemotronBolt.name,
    lfm.name,
  ]);
  assert.ok(ranked.indexOf(embeddingGemma.name) > ranked.indexOf(lfm.name));
  assert.equal(ranked.at(-1), retouched.name);
  assert.ok(freshScore(quietNew, NOW) > 0);
});

test("the Latest feed uses the fresh ranking, which no Hub request ever sends", () => {
  assert.match(
    readSrc("features/hub/lib/channels.ts"),
    /id: "unsloth-latest",[\s\S]*?sort: "fresh",/,
  );
  const hook = readSrc("features/hub/hooks/use-hub-model-search.ts");
  assert.match(
    hook,
    /if \(opts\.sortBy === "fresh"\) \{\n\s*return freshChannelIterator\(channel, opts\);/,
  );
  assert.match(
    hook,
    /url\.searchParams\.set\("sort", hubListingSort\(sortBy\)\)/,
  );
  assert.doesNotMatch(hook, /sort: (opts\.)?sortBy,/);
});

test("Latest picked from the sort menu, outside any section, ranks the same way", () => {
  const hook = readSrc("features/hub/hooks/use-hub-model-search.ts");
  assert.match(
    hook,
    /if \(sortBy === "fresh" && !trimmed\) \{[\s\S]*?freshChannelIterator\(\n\s*\{ owner: unslothOnly \? "unsloth" : undefined \},[\s\S]*?tasks: normalizeTaskFilter\(task\),/,
  );
  assert.match(
    hook,
    /const poolSize = FRESH_POOL \* Math\.max\(1, tasks\.length\);/,
  );
  // The task scopes both candidate listings, not only the ranked result.
  assert.match(hook, /\.\.\.\(opts\.task \? \{ task: opts\.task \} : \{\}\),/);
});

test("datasets and ModelScope never offer Latest", () => {
  const toolbar = readSrc("features/hub/catalog/models-toolbar.tsx");
  assert.match(
    toolbar,
    /!\(option\.value === "fresh" && \(isDataset \|\| hubSource === "modelscope"\)\)/,
  );
  assert.match(
    readSrc("features/hub/hub-page.tsx"),
    /isDatasetMode && requestedSort === "fresh"\n\s*\? "createdAt"/,
  );
});
