// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { RecipePayload } from "./types";

export function createEmptyRecipePayload(): RecipePayload {
  return {
    recipe: {
      model_providers: [],
      mcp_providers: [],
      model_configs: [],
      tool_configs: [],
      columns: [],
      processors: [],
    },
    run: {
      rows: 5,
      preview: true,
      output_formats: ["jsonl"],
    },
    ui: {
      nodes: [],
      edges: [],
      layout_direction: "LR",
    },
  };
}
