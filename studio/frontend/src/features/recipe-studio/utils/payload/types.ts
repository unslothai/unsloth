// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type RecipePayload = {
  recipe: {
    // biome-ignore lint/style/useNamingConvention: api schema
    model_providers: Record<string, unknown>[];
    // biome-ignore lint/style/useNamingConvention: api schema
    mcp_providers: Record<string, unknown>[];
    // biome-ignore lint/style/useNamingConvention: api schema
    model_configs: Record<string, unknown>[];
    // biome-ignore lint/style/useNamingConvention: api schema
    seed_config?: Record<string, unknown>;
    // biome-ignore lint/style/useNamingConvention: api schema
    tool_configs: Record<string, unknown>[];
    columns: Record<string, unknown>[];
    processors: Record<string, unknown>[];
  };
  run: {
    rows: number;
    preview: boolean;
    // biome-ignore lint/style/useNamingConvention: api schema
    output_formats: string[];
    // biome-ignore lint/style/useNamingConvention: backend schema
    execution_type?: "preview" | "full";
    // biome-ignore lint/style/useNamingConvention: backend schema
    run_config?: Record<string, unknown>;
    // biome-ignore lint/style/useNamingConvention: backend schema
    dataset_name?: string;
    // biome-ignore lint/style/useNamingConvention: backend schema
    artifact_path?: string;
    // biome-ignore lint/style/useNamingConvention: backend schema
    merge_batches?: boolean;
    // biome-ignore lint/style/useNamingConvention: backend schema
    run_name?: string | null;
  };
  ui: {
    nodes: Array<{
      id: string;
      x: number;
      y: number;
      width?: number;
      node_type?: "markdown_note" | "tool_config";
      name?: string;
      markdown?: string;
      note_color?: string;
      note_opacity?: string;
      tools_by_provider?: Record<string, string[]>;
    }>;
    edges: {
      from: string;
      to: string;
      type?: string;
      source_handle?: string;
      target_handle?: string;
    }[];
    // ui-only: graph orientation
    layout_direction?: "LR" | "TB";
    // ui-only, used to preserve seed block mode across imports/refresh
    seed_source_type?: "hf" | "local" | "unstructured" | "github_repo";
    // ui-only, persisted aux node positions by llm name + aux key
    aux_nodes?: Array<{
      llm: string;
      key: string;
      x: number;
      y: number;
    }>;
    // ui-only, seed metadata cached for refresh/import UX
    seed_columns?: string[];
    seed_drop_columns?: string[];
    seed_preview_rows?: Record<string, unknown>[];
    local_file_name?: string;
    // biome-ignore lint/style/useNamingConvention: api schema
    unstructured_file_ids?: string[];
    // biome-ignore lint/style/useNamingConvention: api schema
    unstructured_file_names?: string[];
    // biome-ignore lint/style/useNamingConvention: api schema
    unstructured_file_sizes?: number[];
    // biome-ignore lint/style/useNamingConvention: api schema
    unstructured_chunk_size?: string;
    // biome-ignore lint/style/useNamingConvention: api schema
    unstructured_chunk_overlap?: string;
    // ui-only: per-node advanced accordion state
    advanced_open_by_node?: Record<string, boolean>;
  };
};

export type RecipePayloadResult = {
  errors: string[];
  payload: RecipePayload;
};
