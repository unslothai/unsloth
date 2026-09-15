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
    isGguf: false,
  });
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
    isGguf: false,
    tags: ["text-generation", "license:apache-2.0", "en"],
  });
  assert.equal(meta.license, "apache-2.0");
});

test("a repo with no licence tag reports none", () => {
  const meta = metaFromHfResult({
    id: "x/y",
    downloads: 0,
    likes: 0,
    isGguf: false,
    tags: ["text-generation"],
  });
  assert.equal(meta.license, null);
});

test("languages are read out of the tag list", () => {
  const meta = metaFromHfResult({
    id: "x/y",
    downloads: 0,
    likes: 0,
    isGguf: false,
    tags: ["language:en", "language:de", "text-generation"],
  });
  assert.deepEqual(meta.languages, ["en", "de"]);
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
      isGguf: false,
      gated,
    });
    assert.equal(meta.gated, true, gated);
  }
  const open = metaFromHfResult({
    id: "x/y",
    downloads: 0,
    likes: 0,
    isGguf: false,
    gated: false,
  });
  assert.equal(open.gated, false);
});

test("private maps to isPrivate", () => {
  const meta = metaFromHfResult({
    id: "x/y",
    downloads: 0,
    likes: 0,
    isGguf: false,
    private: true,
  });
  assert.equal(meta.isPrivate, true);
});

// downloadsAllTime is the more meaningful number when present: `downloads` is a 30-day
// window, so a long-lived repo otherwise looks quieter than it is.
test("all-time downloads win over the 30-day count", () => {
  const meta = metaFromHfResult({
    id: "x/y",
    downloads: 100,
    likes: 0,
    isGguf: false,
    downloadsAllTime: 5000,
  });
  assert.equal(meta.downloads, 5000);
});

test("a null result yields no facts rather than throwing", () => {
  assert.doesNotThrow(() => metaFromHfResult(null));
  assert.equal(metaFromHfResult(null), null);
});
