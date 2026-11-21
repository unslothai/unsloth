// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The model picker's Recommended list ranks like the Model hub's Latest feed by default.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const pickers = readSrc("features/model-picker/components/model-selector/pickers.tsx");

test("Recommended defaults to the hub's Latest ranking, with Trending and Recent still offered", () => {
  assert.match(pickers, /type RecommendedSortKey = "fresh" \| "trendingScore" \| "lastModified";/);
  assert.match(pickers, /\{ value: "fresh", label: "Latest" \},\n\s*\{ value: "trendingScore", label: "Trending" \},/);
  assert.match(pickers, /useState<RecommendedSortKey>\("fresh"\)/);
});

test("the search-only hook stays idle until there is a query, so browse fetches Latest once", () => {
  assert.match(
    pickers,
    /enabled: online && section === "recommended" && debouncedQuery\.trim\(\)\.length > 0,\n\s*\}\);\n\s*const recommendedSearch = useHubModelSearch\(""/,
  );
});
