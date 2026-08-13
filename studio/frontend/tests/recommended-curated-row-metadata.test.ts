// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Recommended list paints curated catalog seeds as well as live Hub listing rows.
// Everything a row shows beyond its id used to come from the listing alone, so a curated
// model the listing does not return (a repo it has not indexed, a non-unsloth owner, one
// this account cannot see) rendered bare and, once downloaded, stopped matching search
// entirely. These pin the catalog fallbacks that close both gaps.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import ts from "typescript";

import { detectCapabilities } from "../src/features/model-picker/components/model-selector/model-capabilities.ts";
import {
  IMAGE_CATALOG,
  VIDEO_CATALOG,
  curatedCapabilitiesFor,
  curatedTotalParamsFor,
} from "../src/features/model-picker/components/model-selector/model-catalog.ts";
import {
  paramsFromId,
  searchRowFitsDevice,
  searchableRecommendedIds,
} from "../src/features/model-picker/components/model-selector/recommended-fit.ts";

const H3_GGUF = "unsloth/MiniMax-H3-GGUF";
const H3_BF16 = "MiniMaxAI/MiniMax-H3";
const LTX_GGUF = "unsloth/LTX-2.3-GGUF";
const WAN_BF16 = "Wan-AI/Wan2.2-TI2V-5B-Diffusers";

// ── search: a downloaded curated row stays findable ───────────────────────────

test("a seed the listing pool dropped is still searchable", () => {
  // recommendedIds (the listing pool) filters out everything already on disk, because a
  // downloaded model has its own On Device row. The unfiltered Recommended list keeps
  // painting it from the seeds, so search has to as well.
  const seeds = [H3_GGUF, LTX_GGUF];
  const listing = [LTX_GGUF, "unsloth/Wan2.2-TI2V-5B-GGUF"]; // H3 downloaded -> dropped
  assert.deepEqual(searchableRecommendedIds(seeds, listing), [
    H3_GGUF,
    LTX_GGUF,
    "unsloth/Wan2.2-TI2V-5B-GGUF",
  ]);
});

test("seeds come first and no id is listed twice", () => {
  // Same order orderRecommendedRows renders: curated in catalog order, then the rest.
  const out = searchableRecommendedIds(
    [H3_GGUF, LTX_GGUF],
    ["unsloth/Other-GGUF", LTX_GGUF, H3_GGUF],
  );
  assert.deepEqual(out, [H3_GGUF, LTX_GGUF, "unsloth/Other-GGUF"]);
});

test("a listing row that only differs in case does not duplicate its seed", () => {
  // The HF cache lowercases repo ids, so the two pools can disagree on casing.
  const out = searchableRecommendedIds([H3_GGUF], ["unsloth/minimax-h3-gguf"]);
  assert.deepEqual(out, [H3_GGUF]);
});

test("with no seeds the listing pool is passed through unchanged", () => {
  // Chat (no catalog) must behave exactly as before.
  const listing = ["unsloth/a-GGUF", "unsloth/b-GGUF"];
  assert.deepEqual(searchableRecommendedIds([], listing), listing);
});

// ── row metadata: size chip + capability glyphs ───────────────────────────────

test("a curated GGUF carries the param count its id cannot spell", () => {
  // "MiniMax-H3-GGUF" has no "<n>B" token, and the listing does not return the repo, so
  // without the catalog the row has no size chip at all.
  assert.equal(paramsFromId(H3_GGUF), undefined);
  assert.equal(curatedTotalParamsFor(H3_GGUF, VIDEO_CATALOG), 20_111_438_744);
});

test("the curated param count is the artifact's own, not the group's", () => {
  // The BF16 pipeline row is a different checkpoint (it bundles the encoder and VAEs);
  // it declares no count rather than borrowing the denoiser's.
  assert.equal(curatedTotalParamsFor(H3_BF16, VIDEO_CATALOG), undefined);
  assert.equal(curatedTotalParamsFor(LTX_GGUF, VIDEO_CATALOG), 21_005_004_544);
});

test("a curated audio family reports audio the repo name never mentions", () => {
  assert.equal(detectCapabilities({ id: H3_GGUF }).audio, false);
  assert.equal(curatedCapabilitiesFor(H3_GGUF, VIDEO_CATALOG)?.audio, true);
  // Group-level, so every artifact of the model agrees.
  assert.equal(curatedCapabilitiesFor(H3_BF16, VIDEO_CATALOG)?.audio, true);
  assert.equal(curatedCapabilitiesFor(LTX_GGUF, VIDEO_CATALOG)?.audio, true);
});

test("curated capabilities claim nothing they were not given, beyond the scope", () => {
  const caps = curatedCapabilitiesFor(H3_GGUF, VIDEO_CATALOG);
  // vision and reasoning stay false: nothing declared them, and nothing may infer them.
  assert.deepEqual(caps, {
    vision: false,
    reasoning: false,
    audio: true,
    imageGen: false,
    // Not a declaration but not a guess either: the group sits in the video catalog.
    videoGen: true,
  });
  // A group declaring no capabilities still answers, because its scope alone says what it makes.
  // Undefined is reserved for an id the catalog does not know.
  assert.deepEqual(curatedCapabilitiesFor(WAN_BF16, VIDEO_CATALOG), {
    vision: false,
    reasoning: false,
    audio: false,
    imageGen: false,
    videoGen: true,
  });
  assert.equal(curatedCapabilitiesFor("someone/not-curated", VIDEO_CATALOG), undefined);
});

test("an image group reports image generation, not video", () => {
  const caps = curatedCapabilitiesFor("Qwen/Qwen-Image-2512", IMAGE_CATALOG);
  assert.equal(caps?.imageGen, true);
  assert.equal(caps?.videoGen, false);
});

test("every video group whose description says audio declares the capability", () => {
  // The catalog description is the human-facing claim; the flag is what draws the glyph.
  // A new audio family that sets one and forgets the other is the failure this catches.
  for (const group of VIDEO_CATALOG) {
    const saysAudio = /\baudio\b/i.test(group.description);
    assert.equal(
      group.capabilities?.audio === true,
      saysAudio,
      `${group.canonicalId}: description "${group.description}" and capabilities.audio disagree`,
    );
  }
});

test("the curated param count is what makes a curated row sizable in search", () => {
  // The search fit check hides anything it cannot size (`requireKnown`), while the
  // unfiltered Recommended list judges the seed row, which carries its own metadata.
  // Without the catalog fallback the same model is kept in one list and hidden in the
  // other the moment the "Fits on device" toggle is on.
  const gpu = {
    available: true,
    memoryTotalGb: 80,
    maxDeviceMemoryGb: 80,
    loadDeviceMemoryGb: 80,
    systemRamAvailableGb: 0,
    budgetKnown: true,
  };
  const opts = { isGguf: true, gpu, inferenceGpu: gpu, taskScoped: true };
  assert.equal(searchRowFitsDevice({ id: H3_GGUF }, opts), false);
  assert.equal(
    searchRowFitsDevice(
      {
        id: H3_GGUF,
        totalParams: curatedTotalParamsFor(H3_GGUF, VIDEO_CATALOG),
      },
      opts,
    ),
    true,
  );
});

// ── wiring: the picker actually reads the fallbacks ───────────────────────────

const PICKERS = fileURLToPath(
  new URL(
    "../src/features/model-picker/components/model-selector/pickers.tsx",
    import.meta.url,
  ),
);

/** Source text of the top-level `const <name> = ...` initializer in pickers.tsx. */
function declarationText(name: string): string {
  const source = ts.createSourceFile(
    PICKERS,
    readFileSync(PICKERS, "utf8"),
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TSX,
  );
  let found: string | null = null;
  const walk = (node: ts.Node): void => {
    if (
      found === null &&
      ts.isVariableDeclaration(node) &&
      ts.isIdentifier(node.name) &&
      node.name.text === name &&
      node.initializer
    ) {
      found = node.initializer.getText(source);
      return;
    }
    ts.forEachChild(node, walk);
  };
  walk(source);
  assert.ok(found, `no const ${name} = ... in pickers.tsx`);
  return found as unknown as string;
}

test("the search list is built from the seeds as well as the listing pool", () => {
  assert.match(
    declarationText("filteredRecommendedIds"),
    /searchableRecommendedIds\(\s*catalogSeedIds\s*,\s*recommendedIds\s*\)/,
  );
});

test("row meta falls back to the curated seeds", () => {
  const text = declarationText("recommendedMeta");
  // Ordered: seeds behind the listing, community last. A community row ahead of the seeds
  // would let a metadata-poor listing row shadow a curated one, and the map keeps the first.
  assert.match(
    text,
    /recommendedSearch\.results\s*,\s*\.\.\.catalogSeedRows\s*,\s*\.\.\.communityBrowse\.results/,
  );
  // Listing first, and the first entry per id wins.
  assert.match(text, /if\s*\(map\.has\(r\.id\)\)\s*continue;/);
});

test("every pipeline tag the Images picker lists reads as image generation", () => {
  // A row the picker lists has to draw the glyph whatever its repo is called, so the tags
  // detectCapabilities knows cannot be a subset of the ones the picker filters on.
  const tags = declarationText("IMAGE_GEN_TASKS").match(/"([^"]+)"/g) ?? [];
  assert.ok(tags.length > 0, "no tags parsed out of IMAGE_GEN_TASKS");
  for (const quoted of tags) {
    const tag = quoted.slice(1, -1);
    assert.equal(
      detectCapabilities({ id: "someone/unfamiliar-name", pipelineTag: tag }).imageGen,
      true,
      `${tag} is listed by the Images picker but does not read as image generation`,
    );
  }
});

test("a curated pipeline's fit comes from its size, not the QLoRA estimator", () => {
  const text = declarationText("recommendedMeta");
  // The curated size has to be consulted BEFORE estimateLoadingVram, which assumes a language
  // model it can 4-bit quantize and reads the 30 GB Wan 2.2 TI2V pipeline as 5.9 GB.
  const curatedAt = text.indexOf("curatedSizeBytes");
  const estimatorAt = text.indexOf("estimateLoadingVram");
  assert.ok(curatedAt >= 0, "recommendedMeta ignores the curated size");
  assert.ok(estimatorAt >= 0, "recommendedMeta no longer estimates VRAM at all");
  assert.ok(curatedAt < estimatorAt, "the QLoRA estimator runs before the curated size");
});

test("a curated pipeline's fit is judged on the device it loads into", () => {
  const text = declarationText("recommendedMeta");
  // A task load puts the whole pipeline on one device, and torch is not the GGUF backend's
  // inventory, so the badge has to use the same budget the list's own fit filter uses.
  assert.match(text, /loadScopedGpu\(gpu, Boolean\(task\)\)/);
  assert.match(text, /exceedsSize\(curatedBytes, pipelineBudget\)/);
  // GGUF rows keep the inference backend's budget, which is what they load through.
  assert.match(text, /exceedsSize\(sizeBytes, inferenceGpu\)/);
});

test("capabilities fall back to the curated catalog", () => {
  assert.match(
    declarationText("capsById"),
    /curatedCapabilitiesFor\(row\.id, catalog\)/,
  );
});

test("the search fit check falls back to the curated param count", () => {
  assert.match(
    declarationText("searchRowFits"),
    /curatedTotalParamsFor\(row\.id, catalog\)/,
  );
});

test("seed rows carry the curated param count", () => {
  assert.match(
    declarationText("catalogSeedRows"),
    /totalParams:\s*catalog\s*\?\s*curatedTotalParamsFor\(id, catalog\)/,
  );
});
