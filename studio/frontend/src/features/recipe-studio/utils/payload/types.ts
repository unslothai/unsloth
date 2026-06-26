// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type RecipePayload = {
  recipe: {
    model_providers: Record<string, unknown>[];
    mcp_providers: Record<string, unknown>[];
    model_configs: Record<string, unknown>[];
    seed_config?: Record<string, unknown>;
    tool_configs: Record<string, unknown>[];
    columns: Record<string, unknown>[];
    processors: Record<string, unknown>[];
    evaluations: Record<string, unknown>[];
  };
  run: {
    rows: number;
    preview: boolean;
    output_formats: string[];
    execution_type?: "preview" | "full";
    run_config?: Record<string, unknown>;
    dataset_name?: string;
    artifact_path?: string;
    merge_batches?: boolean;
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
    unstructured_file_ids?: string[];
    unstructured_file_names?: string[];
    unstructured_file_sizes?: number[];
    unstructured_chunk_size?: string;
    unstructured_chunk_overlap?: string;
    // ui-only: per-node advanced accordion state
    advanced_open_by_node?: Record<string, boolean>;
  };
};

export type RecipePayloadResult = {
  errors: string[];
  payload: RecipePayload;
};
