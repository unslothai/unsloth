// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { importRecipePayload } = await import(
  "../src/features/recipe-studio/utils/import/importer.ts"
);
const { buildRecipePayload } = await import(
  "../src/features/recipe-studio/utils/payload/build-payload.ts"
);

function rebuiltProviderType(providerType: string): unknown {
  const recipe = {
    recipe: {
      model_providers: [
        {
          name: "local",
          endpoint: "http://localhost:8000/v1",
          provider_type: "openai",
        },
      ],
      model_configs: [{ alias: "writer", model: "m", provider: "local" }],
      mcp_providers: [
        {
          name: "docs",
          provider_type: providerType,
          endpoint: "http://localhost:8931/sse",
        },
      ],
      tool_configs: [{ tool_alias: "docs-tools", providers: ["docs"] }],
      columns: [
        {
          column_type: "llm-text",
          name: "answer",
          model_alias: "writer",
          prompt: "Answer a question.",
          tool_alias: "docs-tools",
        },
      ],
    },
  };
  const imported = importRecipePayload(JSON.stringify(recipe));
  assert.deepEqual(imported.errors, []);
  assert.ok(imported.snapshot);
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
  const providers = built.payload.recipe.mcp_providers as Record<
    string,
    unknown
  >[];
  assert.equal(providers.length, 1);
  return providers[0].provider_type;
}

test("an imported MCP provider keeps its transport in the run payload", () => {
  for (const providerType of ["sse", "streamable_http"]) {
    assert.equal(rebuiltProviderType(providerType), providerType);
  }
});
