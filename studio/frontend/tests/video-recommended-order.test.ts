// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { HostClass } from "../src/features/model-picker/components/model-selector/host-artifact-policy.ts";
import {
  AUDIO_CATALOG,
  type CatalogGroup,
  IMAGE_CATALOG,
  VIDEO_CATALOG,
  catalogToModelOptions,
  curatedRowLabelFor,
  groupForRepoId,
  videoRecommendedTierFor,
} from "../src/features/model-picker/components/model-selector/model-catalog.ts";
import { orderRecommendedRows } from "../src/features/model-picker/components/model-selector/recommended-fit.ts";

const HOSTS: HostClass[] = ["unknown", "gguf-only", "cpu-only", "accelerated", "dense-quant"];
const FP8 = ["fp8"];

type Row = { id: string; isGguf?: boolean };

/** The Recommended list as pickers.tsx builds it: catalog seeds, then the unsloth/* listing. */
function recommended(
  catalog: CatalogGroup[],
  host: HostClass,
  schemes: readonly string[],
  listing: readonly string[],
  tiers = true,
): string[] {
  const seeds: Row[] = catalogToModelOptions(catalog, host, schemes).map((m) => ({
    id: m.id,
    isGguf: Boolean(m.isGguf),
  }));
  const results: Row[] = listing.map((id) => ({ id, isGguf: /-gguf$/i.test(id) }));
  return orderRecommendedRows({
    seeds,
    results,
    keep: () => true,
    deviceFiltered: false,
    fits: () => true,
    familyOf: (id) => groupForRepoId(id, catalog)?.canonicalId.toLowerCase(),
    pinnedFamilies: catalog.filter((g) => g.pinToTop).map((g) => g.canonicalId.toLowerCase()),
    pinnedTogether: catalog
      .filter((g) => g.pinToTop && g.pinRowsTogether)
      .map((g) => g.canonicalId.toLowerCase()),
    tierOf: tiers ? videoRecommendedTierFor(catalog, host, schemes) : undefined,
  }).map((r) => r.id);
}

// The trending listing from the report: LTX-2.5's GGUF first.
const VIDEO_LISTING = ["unsloth/LTX-2.5-GGUF", "unsloth/MiniMax-H3-GGUF", "unsloth/LTX-2.3-GGUF"];

const names = (ids: string[], host: HostClass, schemes: readonly string[]) =>
  ids.map((id) => curatedRowLabelFor(id, VIDEO_CATALOG, host, schemes)?.name);

test("video on a dense-quant host: Fast FP8 rows lead, Slow GGUF rows trail", () => {
  const out = names(recommended(VIDEO_CATALOG, "dense-quant", FP8, VIDEO_LISTING), "dense-quant", FP8);
  assert.deepEqual(out.slice(0, 2), ["MiniMax H3 (Fast FP8)", "LTX 2.5 distilled (Diffusers) (Fast FP8)"]);
  const firstSlow = out.findIndex((n) => n?.includes("(Slow)"));
  assert.ok(firstSlow > 2);
  assert.ok(out.slice(firstSlow).every((n) => n?.includes("(Slow)")), out.join(" | "));
  assert.deepEqual(out.slice(firstSlow), [
    "LTX-2.5-GGUF (Slow)",
    "MiniMax-H3-GGUF (Slow)",
    "LTX-2.3-GGUF (Slow)",
  ]);
  assert.equal(out[2], "LTX 2.5 distilled (Single file)");
});

test("video on an accelerated host without dense quant: the Fast H3 row leads, GGUFs trail", () => {
  const out = names(recommended(VIDEO_CATALOG, "accelerated", [], VIDEO_LISTING), "accelerated", []);
  assert.equal(out[0], "MiniMax H3 (Fast)");
  const firstSlow = out.findIndex((n) => n?.includes("(Slow)"));
  assert.ok(out.slice(firstSlow).every((n) => n?.includes("(Slow)")), out.join(" | "));
  assert.ok(out.slice(0, firstSlow).every((n) => !n?.includes("(Slow)")));
});

test("the lead rows keep catalog order whatever the listing trends; the rest follow the listing", () => {
  for (const listing of [VIDEO_LISTING, [...VIDEO_LISTING].reverse(), []]) {
    const out = recommended(VIDEO_CATALOG, "dense-quant", FP8, listing);
    assert.deepEqual(out.slice(0, 2), ["MiniMaxAI/MiniMax-H3", "Lightricks/LTX-2.5-Diffusers"]);
    assert.ok(out.slice(-3).every((id) => /-gguf$/i.test(id)), out.join(" | "));
  }
  assert.equal(recommended(VIDEO_CATALOG, "dense-quant", FP8, [...VIDEO_LISTING].reverse())[2], "Lightricks/LTX-2.3");
});

test("hosts without the Fast / Slow split keep the untiered video order", () => {
  for (const host of ["unknown", "gguf-only", "cpu-only"] as HostClass[]) {
    assert.equal(videoRecommendedTierFor(VIDEO_CATALOG, host, FP8), undefined);
    assert.deepEqual(
      recommended(VIDEO_CATALOG, host, FP8, VIDEO_LISTING),
      recommended(VIDEO_CATALOG, host, FP8, VIDEO_LISTING, false),
    );
  }
});

test("Images and Audio get no tiers on any host", () => {
  for (const catalog of [IMAGE_CATALOG, AUDIO_CATALOG]) {
    for (const host of HOSTS) {
      for (const schemes of [[], FP8]) {
        assert.equal(videoRecommendedTierFor(catalog, host, schemes), undefined);
      }
    }
  }
  assert.equal(videoRecommendedTierFor(undefined, "dense-quant", FP8), undefined);
});
