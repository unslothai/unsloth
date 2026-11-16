// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Searching a model maker by name also lists Unsloth's uploads of that maker's models.

import assert from "node:assert/strict";
import test from "node:test";

import {
  matchProviderByQuery,
  providerHubOrgs,
  providerSearchTerms,
} from "../src/features/hub/lib/provider-search.ts";
import { parseOwnerSearchQuery } from "../src/features/hub/lib/search-text.ts";
import ts from "@typescript/typescript6";
import { readSrc } from "./helpers/kit.ts";

test("owner/model searches stay scoped, including an owner-only trailing slash", () => {
  assert.deepEqual(parseOwnerSearchQuery("Atomic-Germ/Qwen"), {
    owner: "Atomic-Germ",
    query: "Qwen",
  });
  assert.deepEqual(parseOwnerSearchQuery("Atomic-Germ/"), {
    owner: "Atomic-Germ",
    query: "",
  });
  assert.equal(parseOwnerSearchQuery("Qwen"), null);
  assert.equal(parseOwnerSearchQuery("https://huggingface.co/Atomic-Germ/Qwen"), null);
});

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
  assert.match(hook, /if \(publisher && !unslothOnly && !channelOwner && !channelQuery\) \{\s*return ownerModelIterator\(/);
  // A tag-only channel returns early, so the owner branch must come first.
  assert.ok(hook.indexOf("return ownerModelIterator(") < hook.indexOf("if (channelOwner || channelTagsKey || channelQuery)"));
  assert.equal(hook.match(/providerUnslothFirstIterator\(queryProvider, /g)?.length, 2);
  assert.match(hook, /resolveOwnerProviderLogo\("unsloth", name\.split\("\/"\)\[1\]\)\?\.id === provider\.id/);
});

test("a maker search also reaches the maker's own Hub orgs, outside Unsloth-only scope", () => {
  assert.deepEqual(providerHubOrgs(matchProviderByQuery("alibaba")!), ["Qwen"]);
  assert.deepEqual(providerHubOrgs(matchProviderByQuery("meta")!), ["meta-models", "meta-llama", "facebook"]);
  const hook = readSrc("features/hub/hooks/use-hub-model-search.ts");
  assert.match(hook, /providerUnslothFirstIterator\(queryProvider, orgSearch, \{\n\s*\.\.\.providerOpts,\n\s*includeOrgs: false,/);
  assert.match(hook, /providerUnslothFirstIterator\(queryProvider, merged, \{\n\s*\.\.\.providerOpts,\n\s*includeOrgs: true,/);
});

test("the maker groups stream: Unsloth first, listing tails kept, a rejecting filter bounded", () => {
  const hook = readSrc("features/hub/hooks/use-hub-model-search.ts");
  const fn = hook.slice(hook.indexOf("async function* providerUnslothFirstIterator("), hook.indexOf("const FRESH_POOL"));
  // The org listings start with the Unsloth ones but are awaited only after the Unsloth group yields.
  assert.match(fn, /const orgFirst = Promise\.all\(orgIters\.map\(take\)\);\n[\s\S]*const unslothFirst = await Promise\.all/);
  assert.match(fn, /yield\* ranked\(unslothFirst\.flat\(\), isProviders\);\n\s*yield\* ranked\(\(await orgFirst\)\.flat\(\)/);
  // Pools past the first page are not dropped.
  assert.match(fn, /yield\* tail\(unslothIters, isProviders\);\n\s*yield\* tail\(orgIters, \(\) => true\);/);
  assert.match(fn, /if \(\+\+rejected >= UNSLOTH_FILTER_REJECT_LIMIT\) break;/);
  assert.equal(hook.match(/includeOrgs: (true|false),\n\s*filterPriority,/g)?.length, 2);
});

function ownerIterator(repos: string[], calls: unknown[]) {
  const hook = readSrc("features/hub/hooks/use-hub-model-search.ts");
  const start = hook.indexOf("async function* ownerModelIterator(");
  const source = hook.slice(start, hook.indexOf("\n}\n", start) + 2);
  const listModels = async function* (args: { search: { owner?: string; query?: string } }) {
    calls.push(args.search);
    const { owner, query = "" } = args.search;
    for (const name of repos) {
      const [o, n] = name.split("/");
      if (owner ? o === owner && n.includes(query) : name.toLowerCase().includes(query.toLowerCase())) yield { name };
    }
  };
  const modelInfo = async ({ name }: { name: string }) => {
    const hit = repos.find((repo) => repo.toLowerCase() === name.toLowerCase());
    if (!hit) throw new Error("404");
    return { name: hit };
  };
  return new Function(
    "listModels", "cachedModelInfo", "getHfEndpoint", "makeSortFetch", "makeHfFetch", "primeFromListing", "ALL_FIELDS",
    ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText + "; return ownerModelIterator;",
  )(listModels, modelInfo, () => "", () => fetch, () => fetch, () => {}, []);
}

async function names(it: AsyncGenerator<{ name: string }>) {
  const out: string[] = [];
  for await (const model of it) out.push(model.name);
  return out;
}

test("an exact repo id leads its owner's listing, without a task filter", async () => {
  const calls: unknown[] = [];
  const iter = ownerIterator(["unsloth/R1-Qwen3-8B-GGUF", "unsloth/Qwen3-8B-GGUF", "Qwen/Qwen3-8B"], calls);
  assert.deepEqual(await names(iter("unsloth", "Qwen3-8B-GGUF", undefined, "unsloth/qwen3-8b-gguf")), [
    "unsloth/Qwen3-8B-GGUF",
    "unsloth/R1-Qwen3-8B-GGUF",
  ]);
  assert.deepEqual(calls, [{ owner: "unsloth", query: "Qwen3-8B-GGUF" }]);
  calls.length = 0;
  await names(iter("unsloth", "", undefined, undefined, "downloads", "desc", undefined, "", ["gguf"]));
  assert.deepEqual(calls, [{ owner: "unsloth", tags: ["gguf"] }]);
});

test("a miscased owner retries under the Hub's spelling", async () => {
  const calls: unknown[] = [];
  const iter = ownerIterator(["Atomic-Germ/Qwen3.8-9B", "Atomic-Germ/Gemma4", "xatomic-germ/Qwen"], calls);
  assert.deepEqual(await names(iter("atomic-germ", "Qwen")), ["Atomic-Germ/Qwen3.8-9B"]);
  assert.deepEqual(await names(iter("atomic-germ", "")), ["Atomic-Germ/Qwen3.8-9B", "Atomic-Germ/Gemma4"]);
  assert.deepEqual(await names(iter("nobody", "")), []);
});
