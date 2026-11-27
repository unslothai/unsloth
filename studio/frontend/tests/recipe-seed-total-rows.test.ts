// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { importRecipePayload } = await import(
  "../src/features/recipe-studio/utils/import/importer.ts"
);
const { buildRecipePayload } = await import(
  "../src/features/recipe-studio/utils/payload/build-payload.ts"
);

test("a document seed's total chunk count survives save and reload", () => {
  const template = JSON.parse(
    readFileSync(
      new URL(
        "../src/features/data-recipes/learning-recipes/books-and-documents.json",
        import.meta.url,
      ),
      "utf8",
    ),
  );
  template.ui.seed_total_rows = 42;
  const imported = importRecipePayload(JSON.stringify(template));
  assert.ok(imported.snapshot);
  const seed = Object.values(imported.snapshot.configs).find(
    (config) => config.kind === "seed",
  );
  assert.equal(seed?.kind === "seed" ? seed.seed_total_rows : null, 42);

  const {
    configs,
    nodes,
    edges,
    processors,
    layoutDirection,
    auxNodePositions,
  } = imported.snapshot;
  const built = buildRecipePayload(
    configs,
    nodes,
    edges,
    processors,
    layoutDirection,
    auxNodePositions,
  );
  assert.equal(built.payload.ui.seed_total_rows, 42);
});
