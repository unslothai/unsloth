// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Searching a model maker by name also lists Unsloth's uploads of that maker's models.

import assert from "node:assert/strict";
import test from "node:test";

import { matchProviderByQuery, providerSearchTerms } from "../src/features/hub/lib/provider-search.ts";
import { readSrc } from "./helpers/kit.ts";

test("maker names, orgs and common aliases resolve; partial words do not", () => {
  assert.equal(matchProviderByQuery("Meta")?.id, "meta-llama");
  assert.equal(matchProviderByQuery("meta-llama")?.id, "meta-llama");
  assert.equal(matchProviderByQuery("alibaba")?.id, "qwen");
  assert.equal(matchProviderByQuery("zai")?.id, "zai-org");
  assert.equal(matchProviderByQuery("Z.ai")?.id, "zai-org");
  assert.equal(matchProviderByQuery("google")?.id, "google");
  assert.equal(matchProviderByQuery("NVIDIA")?.id, "nvidia");
  assert.equal(matchProviderByQuery("metal"), null);
  assert.equal(matchProviderByQuery("gem"), null);
  assert.equal(matchProviderByQuery("llama-3"), null);
  assert.equal(matchProviderByQuery("  "), null);
});

test("search terms cover the maker's families without redundant requests", () => {
  const meta = providerSearchTerms(matchProviderByQuery("meta")!);
  assert.ok(meta.includes("muse"));
  assert.ok(meta.includes("llama"));
  assert.ok(!meta.includes("meta-llama"), "contains llama already");
  assert.deepEqual(providerSearchTerms(matchProviderByQuery("google")!).filter((t) => t.includes("gemma")), ["gemma"]);
});

test("both typed-search paths put the maker's Unsloth uploads first, never for owner/repo", () => {
  const hook = readSrc("features/hub/hooks/use-hub-model-search.ts");
  assert.match(hook, /isPublisherQuery \|\| !trimmed \? null : matchProviderByQuery\(trimmed\)/);
  assert.equal(hook.match(/providerUnslothFirstIterator\(queryProvider, /g)?.length, 2);
  assert.match(hook, /resolveOwnerProviderLogo\("unsloth", name\.split\("\/"\)\[1\]\)\?\.id === provider\.id/);
});
