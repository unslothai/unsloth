// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Contract for turning an HF search/info result into the panel's input (issue #11017).
// The adapter is where field names from the API meet the panel's own shape, so these
// cases pin the mappings that a rename upstream would otherwise silently break.

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { metaFromHfResult } = await import(
  "../src/features/model-picker/components/model-selector/model-info-facts.ts"
);

test("core fields map across", () => {
  const meta = metaFromHfResult({
    id: "unsloth/Llama-3.1-8B",
    downloads: 42,
    likes: 7,
    totalParams: 8_030_000_000,
    estimatedSizeBytes: 16_060_000_000,
    libraryName: "transformers",
    pipelineTag: "text-generation",
    createdAt: "2024-07-23T00:00:00.000Z",
    updatedAt: "2025-01-02T00:00:00.000Z",
  });
  assert.ok(meta);
  assert.equal(meta.id, "unsloth/Llama-3.1-8B");
  assert.equal(meta.downloads, 42);
  assert.equal(meta.likes, 7);
  assert.equal(meta.totalParams, 8_030_000_000);
  assert.equal(meta.sizeBytes, 16_060_000_000);
  assert.equal(meta.library, "transformers");
  assert.equal(meta.pipelineTag, "text-generation");
  assert.equal(meta.createdAt, "2024-07-23T00:00:00.000Z");
  assert.equal(meta.lastModified, "2025-01-02T00:00:00.000Z");
});

// HF carries the licence as a `license:` tag rather than a field, so the panel would show
// "not stated" for every model if this mapping were dropped.
test("licence is read out of the tag list", () => {
  const meta = metaFromHfResult({
    id: "x/y",
    downloads: 0,
    likes: 0,
    tags: ["text-generation", "license:apache-2.0", "en"],
  });
  assert.ok(meta);
  assert.equal(meta.license, "apache-2.0");
});

test("a repo with no licence tag reports none", () => {
  const meta = metaFromHfResult({
    id: "x/y",
    downloads: 0,
    likes: 0,
    tags: ["text-generation"],
  });
  assert.ok(meta);
  assert.equal(meta.license, null);
});

test("languages are read out of the tag list", () => {
  const meta = metaFromHfResult({
    id: "x/y",
    downloads: 0,
    likes: 0,
    tags: ["language:en", "language:de", "text-generation"],
  });
  assert.ok(meta);
  assert.deepEqual(meta.languages, ["en", "de"]);
});

// HF flattens a model card's `language:` frontmatter into bare codes, so most model repos
// carry `en`, not `language:en`. Reading only the prefixed form dropped the Languages row for
// exactly the repos most likely to have one (issue #11033 review).
test("bare Hugging Face language codes are accepted too", () => {
  const meta = metaFromHfResult({
    id: "x/y",
    downloads: 0,
    likes: 0,
    tags: ["en", "de", "text-generation", "gguf"],
  });
  assert.ok(meta);
  assert.deepEqual(meta.languages, ["en", "de"]);
});

test("prefixed and bare codes mix without duplicating", () => {
  const meta = metaFromHfResult({
    id: "x/y",
    downloads: 0,
    likes: 0,
    tags: ["language:en", "en", "fr"],
  });
  assert.ok(meta);
  assert.deepEqual(meta.languages, ["en", "fr"]);
});

// A bare code cannot be recognised by shape alone, so matching is against the ISO 639-1 set
// rather than tag length: "rl", "4-bit" and "gguf" are not languages and must not appear.
//
// The set cannot disambiguate every case — a bare "ml" is Malayalam's ISO code and also how a
// repo might tag "machine learning". It resolves to the language, which is what the tag means
// in HF's language vocabulary; the cost of being wrong is one stray chip, against dropping the
// Languages row entirely for the many repos that tag bare codes.
test("short non-language tags are not mistaken for languages", () => {
  const meta = metaFromHfResult({
    id: "x/y",
    downloads: 0,
    likes: 0,
    tags: ["rl", "4-bit", "gguf", "text-generation", "moe"],
  });
  assert.ok(meta);
  assert.equal(meta.languages, undefined);
});

// The region is information the reader wants, so the tag keeps its own spelling even though
// only the base code decides whether it counts as a language at all.
test("region-qualified codes keep their full spelling", () => {
  const meta = metaFromHfResult({
    id: "x/y",
    downloads: 0,
    likes: 0,
    tags: ["pt-br", "zh-CN"],
  });
  assert.ok(meta);
  assert.deepEqual(meta.languages, ["pt-br", "zh-CN"]);
});

// `gated` is `false | "auto" | "manual"` upstream. Only `false` means ungated, so a
// truthiness check would be right by accident; an equality check against "manual" would
// miss "auto" and tell the user an auto-gated repo downloads freely.
test("both gating modes count as gated", () => {
  for (const gated of ["auto", "manual"] as const) {
    const meta = metaFromHfResult({
      id: "x/y",
      downloads: 0,
      likes: 0,
      gated,
    });
  assert.ok(meta);
    assert.equal(meta.gated, true, gated);
  }
  const open = metaFromHfResult({
    id: "x/y",
    downloads: 0,
    likes: 0,
    gated: false,
  });
  assert.ok(open);
  assert.equal(open.gated, false);
});

test("private maps to isPrivate", () => {
  const meta = metaFromHfResult({
    id: "x/y",
    downloads: 0,
    likes: 0,
    private: true,
  });
  assert.ok(meta);
  assert.equal(meta.isPrivate, true);
});

// downloadsAllTime is the more meaningful number when present: `downloads` is a 30-day
// window, so a long-lived repo otherwise looks quieter than it is.
test("all-time downloads win over the 30-day count", () => {
  const meta = metaFromHfResult({
    id: "x/y",
    downloads: 100,
    likes: 0,
    downloadsAllTime: 5000,
  });
  assert.ok(meta);
  assert.equal(meta.downloads, 5000);
});

test("a null result yields no facts rather than throwing", () => {
  assert.doesNotThrow(() => metaFromHfResult(null));
  assert.equal(metaFromHfResult(null), null);
});

// A tag whose text before the first hyphen happens to spell an ISO 639-1 code is not a
// language. `ml-agents` is the Unity toolkit and rides on thousands of Hub repos; matching only
// the base segment put "ML-AGENTS" in the Languages row of every one of them.
test("hyphenated tags that merely start with a language code are not languages", () => {
  for (const tag of ["ml-agents", "mt-bench", "no-code", "or-else"]) {
    const meta = metaFromHfResult({ id: "a/b", tags: [tag] });
    assert.ok(meta);
    assert.equal(meta.languages, undefined, tag);
  }
});

// Region and script subtags are still languages and still keep their own spelling.
test("region- and script-qualified codes are kept", () => {
  const meta = metaFromHfResult({
    id: "a/b",
    tags: ["zh-CN", "pt-br", "sr-Latn"],
  });
  assert.ok(meta);
  assert.deepEqual(meta.languages, ["zh-CN", "pt-br", "sr-Latn"]);
});

// HF flattens a card's `language:` frontmatter into `tags` while often keeping the prefixed
// form too, so the same language arrives twice in two spellings. Listing it as "EN, EN" is the
// panel contradicting itself inside one row.
test("the same language in two spellings is listed once", () => {
  const meta = metaFromHfResult({ id: "a/b", tags: ["language:EN", "en"] });
  assert.ok(meta);
  assert.deepEqual(meta.languages, ["EN"]);
});

// A prefix states intent, not validity: HF cards carry `language:multilingual`, which is not a
// language and has no place among the chips.
test("a prefixed value that is not a language code is rejected", () => {
  const meta = metaFromHfResult({
    id: "a/b",
    tags: ["language:multilingual", "language:en"],
  });
  assert.ok(meta);
  assert.deepEqual(meta.languages, ["en"]);
});

// `recommended-fit.ts` states the rule and the reason: the curated size outranks the estimate,
// because `estimatedSizeBytes` is the full-precision checkpoint. Reading it the other way made
// this panel quote a number up to four times the download while the size badge on the same row
// quoted the quantized load.
test("the curated size outranks the full-precision estimate", () => {
  const meta = metaFromHfResult({
    id: "unsloth/Llama-3.1-8B-Instruct",
    estimatedSizeBytes: 16_060_522_496,
    curatedSizeBytes: 4_900_000_000,
  });
  assert.ok(meta);
  assert.equal(meta.sizeBytes, 4_900_000_000);
  assert.equal(meta.sizeIsFullPrecision, false);
});

test("a full-precision estimate is marked as one", () => {
  const meta = metaFromHfResult({
    id: "unsloth/Llama-3.1-8B-Instruct",
    estimatedSizeBytes: 16_060_522_496,
  });
  assert.ok(meta);
  assert.equal(meta.sizeBytes, 16_060_522_496);
  assert.equal(meta.sizeIsFullPrecision, true);
});
