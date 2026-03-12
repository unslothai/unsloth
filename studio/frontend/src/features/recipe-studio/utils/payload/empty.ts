// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { RecipePayload } from "./types";

export function createEmptyRecipePayload(): RecipePayload {
  return {
    recipe: {
      // biome-ignore lint/style/useNamingConvention: api schema
      model_providers: [],
      // biome-ignore lint/style/useNamingConvention: api schema
      mcp_providers: [],
      // biome-ignore lint/style/useNamingConvention: api schema
      model_configs: [],
      // biome-ignore lint/style/useNamingConvention: api schema
      tool_configs: [],
      columns: [],
      processors: [],
    },
    run: {
      rows: 5,
      preview: true,
      // biome-ignore lint/style/useNamingConvention: api schema
      output_formats: ["jsonl"],
    },
    ui: {
      nodes: [],
      edges: [],
      layout_direction: "LR",
    },
  };
}
