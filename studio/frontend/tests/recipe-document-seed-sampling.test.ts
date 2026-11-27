// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { makeSeedConfig, swapSeedSourceType } = await import(
  "../src/features/recipe-studio/utils/config-factories.ts"
);
const { buildSeedConfig } = await import(
  "../src/features/recipe-studio/utils/payload/builders-seed.ts"
);
const { importRecipePayload } = await import(
  "../src/features/recipe-studio/utils/import/importer.ts"
);
const { buildRecipePayload } = await import(
  "../src/features/recipe-studio/utils/payload/build-payload.ts"
);

const RECIPES = new URL(
  "../src/features/data-recipes/learning-recipes/",
  import.meta.url,
);

test("a new documents seed samples chunks from the whole upload", () => {
  const seed = {
    ...makeSeedConfig("s1", [], "unstructured"),
    // biome-ignore lint/style/useNamingConvention: api schema
    hf_path: "/tmp/seed.parquet",
  };
  assert.equal(buildSeedConfig(seed, [])?.sampling_strategy, "shuffle");
  assert.equal(makeSeedConfig("s2", [], "hf").sampling_strategy, "ordered");
});

test("the PDF QA template samples chunks from the whole upload", () => {
  const { snapshot } = importRecipePayload(
    readFileSync(new URL("pdf-grounded-qa.json", RECIPES), "utf8"),
  );
  assert.ok(snapshot);
  const built = buildRecipePayload(
    snapshot.configs,
    snapshot.nodes,
    snapshot.edges,
    snapshot.processors,
    snapshot.layoutDirection,
    snapshot.auxNodePositions,
  );
  assert.equal(built.payload.recipe.seed_config?.sampling_strategy, "shuffle");
});

test("swapping a seed to Documents samples chunks from the whole upload", () => {
  const hf = makeSeedConfig("s1", [], "hf");
  const docs = swapSeedSourceType(hf, "unstructured");
  assert.equal(docs.sampling_strategy, "shuffle");
  assert.equal(swapSeedSourceType(docs, "local").sampling_strategy, "ordered");
  const pinned = { ...docs, sampling_strategy: "ordered" as const };
  assert.equal(
    swapSeedSourceType(pinned, "unstructured").sampling_strategy,
    "ordered",
  );
});
