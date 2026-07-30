// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { recommendedDownloadableGgufVariant } from "../src/features/hub/lib/gguf-recommendation.ts";

const DOWNLOAD_CARD_SOURCE = readFileSync(
  fileURLToPath(
    new URL(
      "../src/features/hub/catalog/gguf-download-card.tsx",
      import.meta.url,
    ),
  ),
  "utf8",
);
const COLLAPSED_RECOMMENDATION =
  /selectedVariantKey === recommendedVariantKey[\s\S]*?recommended/;
const EXPANDED_RECOMMENDATION_PROP =
  /recommended=\{item\.key === recommendedVariantKey\}/;
const EXPANDED_RECOMMENDATION_LABEL = /recommended && \([\s\S]*?recommended/;

type TestVariant = {
  quant: string;
  filename: string;
  downloaded?: boolean;
  partial?: boolean;
};

function variant(
  quant: string,
  overrides: Partial<TestVariant> = {},
): TestVariant {
  return {
    quant,
    filename: `${quant}.gguf`,
    ...overrides,
  };
}

test("recommends the first eligible variant from the fit-sorted list", () => {
  const downloaded = variant("Q8_0", { downloaded: true });
  const bestFit = variant("Q6_K");
  const smallerFit = variant("Q4_K_M");

  assert.equal(
    recommendedDownloadableGgufVariant([downloaded, bestFit, smallerFit]),
    bestFit,
  );
});

test("does not recommend partial variants", () => {
  const partial = variant("Q6_K", { partial: true });
  const completeCandidate = variant("Q4_K_M");

  assert.equal(
    recommendedDownloadableGgufVariant([partial, completeCandidate]),
    completeCandidate,
  );
});

test("falls back to the smallest variant when every candidate exceeds memory", () => {
  const largest = variant("Q8_0");
  const smallest = variant("Q2_K");

  assert.equal(
    recommendedDownloadableGgufVariant([smallest, largest]),
    smallest,
  );
});

test("returns null when every variant is downloaded or partial", () => {
  assert.equal(
    recommendedDownloadableGgufVariant([
      variant("Q6_K", { downloaded: true }),
      variant("Q4_K_M", { partial: true }),
    ]),
    null,
  );
});

test("shows the recommendation in the collapsed variant trigger", () => {
  assert.match(DOWNLOAD_CARD_SOURCE, COLLAPSED_RECOMMENDATION);
});

test("shows the recommendation beside its expanded variant row", () => {
  assert.match(DOWNLOAD_CARD_SOURCE, EXPANDED_RECOMMENDATION_PROP);
  assert.match(DOWNLOAD_CARD_SOURCE, EXPANDED_RECOMMENDATION_LABEL);
});
