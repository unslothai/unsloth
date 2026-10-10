// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Curated seeds the listing does not return need catalog fallbacks for metadata and search.

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

test("a seed the listing pool dropped is still searchable", () => {
  // recommendedIds drops downloaded models, but Recommended still paints them from seeds.
  const seeds = [H3_GGUF, LTX_GGUF];
  const listing = [LTX_GGUF, "unsloth/Wan2.2-TI2V-5B-GGUF"]; // H3 downloaded -> dropped
  assert.deepEqual(searchableRecommendedIds(seeds, listing), [
    H3_GGUF,
    LTX_GGUF,
    "unsloth/Wan2.2-TI2V-5B-GGUF",
  ]);
});

test("seeds come first and no id is listed twice", () => {
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
  const listing = ["unsloth/a-GGUF", "unsloth/b-GGUF"];
  assert.deepEqual(searchableRecommendedIds([], listing), listing);
});

test("a curated GGUF carries the param count its id cannot spell", () => {
  assert.equal(paramsFromId(H3_GGUF), undefined);
  assert.equal(curatedTotalParamsFor(H3_GGUF, VIDEO_CATALOG), 20_111_438_744);
});

test("the curated param count is the artifact's own, not the group's", () => {
  // The BF16 pipeline bundles encoder and VAEs, so it must not borrow the denoiser's count.
  assert.equal(curatedTotalParamsFor(H3_BF16, VIDEO_CATALOG), undefined);
  assert.equal(curatedTotalParamsFor(LTX_GGUF, VIDEO_CATALOG), 21_005_004_544);
});

test("a curated audio family reports audio the repo name never mentions", () => {
  assert.equal(detectCapabilities({ id: H3_GGUF }).audio, false);
  assert.equal(curatedCapabilitiesFor(H3_GGUF, VIDEO_CATALOG)?.audio, true);
  assert.equal(curatedCapabilitiesFor(H3_BF16, VIDEO_CATALOG)?.audio, true);
  assert.equal(curatedCapabilitiesFor(LTX_GGUF, VIDEO_CATALOG)?.audio, true);
});

test("curated capabilities claim nothing they were not given, beyond the scope", () => {
  const caps = curatedCapabilitiesFor(H3_GGUF, VIDEO_CATALOG);
  assert.deepEqual(caps, {
    vision: false,
    reasoning: false,
    audio: true,
    imageGen: false,
    videoGen: true,
  });
  // A group declaring no capabilities still answers from its scope; undefined is for unknown ids.
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
  // Catches a family that sets the description but forgets the glyph flag.
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
  // Search hides unsized rows while Recommended judges the seed, so both need the fallback.
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

const PICKERS = fileURLToPath(
  new URL(
    "../src/features/model-picker/components/model-selector/pickers.tsx",
    import.meta.url,
  ),
);

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
  // The map keeps the first entry, so community rows must come after seeds.
  assert.match(
    text,
    /recommendedSearch\.results\s*,\s*\.\.\.catalogSeedRows\s*,\s*\.\.\.communityBrowse\.results/,
  );
  assert.match(text, /if\s*\(map\.has\(r\.id\)\)\s*continue;/);
});

test("a family name is not read out of a longer word", () => {
  // Family stems run into version digits, so only a following word must not match.
  for (const id of [
    "org/fluxion-7b",
    "org/pixartful-7b",
    "org/ltxtra-2b",
    "org/mochimo-7b",
    "nunchaku/SVDQuant-int4",
  ]) {
    const caps = detectCapabilities({ id });
    assert.equal(caps.imageGen, false, `${id} read as an image generator`);
    assert.equal(caps.videoGen, false, `${id} read as a video generator`);
  }
  assert.equal(detectCapabilities({ id: "org/flux1-dev-fp8" }).imageGen, true);
  assert.equal(detectCapabilities({ id: "stabilityai/sd3.5-large" }).imageGen, true);
  assert.equal(detectCapabilities({ id: "THUDM/CogVideoX-5b" }).videoGen, true);
  assert.equal(detectCapabilities({ id: "stabilityai/svd-xt" }).videoGen, true);
});

test("every pipeline tag the Video picker lists reads as video generation", () => {
  // A row the glyph calls video must route to the Video page, not a chat load.
  const tags = declarationText("VIDEO_GEN_TASKS").match(/"([^"]+)"/g) ?? [];
  assert.ok(tags.length > 0, "no tags parsed out of VIDEO_GEN_TASKS");
  for (const quoted of tags) {
    const tag = quoted.slice(1, -1);
    assert.equal(
      detectCapabilities({ id: "someone/unfamiliar-name", pipelineTag: tag }).videoGen,
      true,
      `${tag} is listed by the Video picker but does not read as video generation`,
    );
  }
});

test("every pipeline tag the Images picker lists reads as image generation", () => {
  // Detected tags cannot be a subset of the picker's filter tags, or listed rows lack glyphs.
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

for (const declaration of ["recommendedMeta", "recommendedVramMap"]) {
  test(`${declaration} asks the catalog about a curated pipeline`, () => {
    const text = declarationText(declaration);
    // estimateLoadingVram assumes a 4-bit-quantizable LM and misreads the 30 GB Wan pipeline as 5.9 GB.
    const curatedAt = text.indexOf("catalogFit(");
    const estimatorAt = text.indexOf("estimateLoadingVram");
    assert.ok(curatedAt >= 0, `${declaration} ignores the curated fit`);
    assert.ok(estimatorAt >= 0, `${declaration} no longer estimates VRAM at all`);
    assert.ok(curatedAt < estimatorAt, "the QLoRA estimator runs first");
    // A task load uses one device and torch's inventory, so the budget must go through loadScopedGpu.
    assert.match(
      text,
      /artifactBudget\(loadScopedGpu\(gpu, Boolean\(task\)\)\)|rowGpu = loadScopedGpu\(gpu, Boolean\(task\)\);[\s\S]{0,400}artifactBudget\(rowGpu\)/,
    );
  });
}

test("every list that judges a row against the device asks the same helper", () => {
  // One verdict for all readers, or a filter calls a row a fit while the badge says OOM.
  assert.match(
    declarationText("catalogFit"),
    /curatedArtifactFit\(id, catalog, budget\)/,
  );
  for (const declaration of ["recommendedRows", "searchRowFits"]) {
    assert.match(
      declarationText(declaration),
      /catalogFit\(/,
      `${declaration} judges rows without the catalog's verdict`,
    );
  }
});

test("GGUF rows keep the inference backend's budget", () => {
  assert.match(
    declarationText("recommendedMeta"),
    /ggufRowFit\(sizeBytes, rowInferenceGpu\)/,
  );
  // Use the inventory of the runtime that places the row, not its file format.
  assert.match(
    declarationText("recommendedMeta"),
    /rowInferenceGpu = diffusionLoad\n\s*\? rowGpu\n\s*: loadScopedGpu\(inferenceGpu, Boolean\(task\)\)/,
  );
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


test("Hub settings configure the cached alias and share the row selection", () => {
  const compiled = ts.transpileModule(
    `const render = ${declarationText("renderHubModelRow")};`,
    { compilerOptions: { jsx: ts.JsxEmit.React, target: ts.ScriptTarget.ES2022 } },
  ).outputText;
  const calls: unknown[][] = [];
  const React = {
    createElement: (type: unknown, props: Record<string, unknown>, ...children: unknown[]) =>
      ({ type, props, children }),
  };
  const render = new Function(
    "React", "isKnownGgufRepo", "onConfigure", "downloadedRowShellClassName",
    "isValueRow", "ROW_ACTIONS_CLASS", "ModelLoadSettingsAction", "cachedIdFor", "pipelineTagById",
    `${compiled}; return render;`,
  )(
    React, () => false, (...args: unknown[]) => calls.push(args),
    (selected: boolean) => selected ? "selected" : "idle", () => true,
    "actions", "settings", (id: string) => id === "mirror/model" ? "vendor/model" : null,
    new Map([["mirror/model", "text-generation"]]),
  );
  const row = render("mirror/model", "model row");
  assert.equal(row.props.className, "selected");
  row.children[1].children[0].props.onConfigure();
  assert.deepEqual(calls[0], ["vendor/model", {
    source: "hub", isLora: false, isGguf: false, isDownloaded: true,
    pipelineTag: "text-generation",
  }]);
  render("new/model", "model row").children[1].children[0].props.onConfigure();
  assert.deepEqual(calls[1], ["new/model", {
    source: "hub", isLora: false, isGguf: false, isDownloaded: false, pipelineTag: null,
  }]);
});
