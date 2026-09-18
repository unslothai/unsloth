// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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

test("hyphenated tags that merely start with a language code are not languages", () => {
  for (const tag of ["ml-agents", "mt-bench", "no-code", "or-else"]) {
    const meta = metaFromHfResult({ id: "a/b", tags: [tag] });
    assert.ok(meta);
    assert.equal(meta.languages, undefined, tag);
  }
});

test("region- and script-qualified codes are kept", () => {
  const meta = metaFromHfResult({
    id: "a/b",
    tags: ["zh-CN", "pt-br", "sr-Latn"],
  });
  assert.ok(meta);
  assert.deepEqual(meta.languages, ["zh-CN", "pt-br", "sr-Latn"]);
});

test("the same language in two spellings is listed once", () => {
  const meta = metaFromHfResult({ id: "a/b", tags: ["language:EN", "en"] });
  assert.ok(meta);
  assert.deepEqual(meta.languages, ["EN"]);
});

test("a prefixed value that is not a language code is rejected", () => {
  const meta = metaFromHfResult({
    id: "a/b",
    tags: ["language:multilingual", "language:en"],
  });
  assert.ok(meta);
  assert.deepEqual(meta.languages, ["en"]);
});

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
