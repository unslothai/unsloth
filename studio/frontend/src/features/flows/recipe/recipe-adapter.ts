// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A Recipe group seen through Recipe Studio's own configs, so its block dialogs, reference sync and payload
// builders edit a flow without the Recipe Studio store. Loaded with the editor dialogs, not with the canvas.

import { syncSubcategoryConfigsForCategoryUpdate } from "@/features/recipe-studio/stores/helpers/edge-sync";
import { applyRenameToConfigs } from "@/features/recipe-studio/stores/helpers/reference-sync";
import type {
  NodeConfig,
  RecipeNode,
  RecipeProcessorConfig,
} from "@/features/recipe-studio/types";
import { importRecipePayload } from "@/features/recipe-studio/utils/import/importer";
import { buildRecipePayload } from "@/features/recipe-studio/utils/payload/build-payload";
import {
  buildExpressionColumn,
  buildLlmColumn,
  buildSamplerColumn,
  buildValidatorColumn,
} from "@/features/recipe-studio/utils/payload/builders";
import type { FlowGroup, FlowNode } from "../types";
import {
  COLUMN_TYPE,
  type RecipeColumn,
  columnName,
  groupPayload,
} from "./recipe-columns";

const COLUMN_KINDS = new Set(["sampler", "llm", "expression", "validator"]);
// Canvas layout from Recipe Studio; a group lays its columns out itself.
const LAYOUT_UI = new Set([
  "nodes",
  "edges",
  "layout_direction",
  "aux_nodes",
  "advanced_open_by_node",
]);

export interface StudioView {
  configs: Record<string, NodeConfig>;
  /** flow node id -> config id, for the column nodes */
  byNode: Record<string, string>;
  processors: RecipeProcessorConfig[];
  errors: string[];
}

/** The group as Recipe Studio configs, the same ones its importer makes for a saved recipe. */
export function studioView(group: FlowGroup, columns: FlowNode[]): StudioView {
  const result = importRecipePayload(
    JSON.stringify(groupPayload(group, columns)),
  );
  if (!result.snapshot) {
    return { configs: {}, byNode: {}, processors: [], errors: result.errors };
  }
  const configs = result.snapshot.configs;
  const idByName = new Map(
    Object.values(configs).map((config) => [config.name, config.id]),
  );
  const byNode: Record<string, string> = {};
  for (const node of columns) {
    const id = idByName.get(columnName(node));
    if (id) byNode[node.id] = id;
  }
  return {
    configs,
    byNode,
    processors: result.snapshot.processors,
    errors: [],
  };
}

/** Recipe Studio's `updateConfig` without the store: the patch plus subcategory, rename and provider sync. */
export function applyConfigPatch(
  configs: Record<string, NodeConfig>,
  id: string,
  patch: Partial<NodeConfig>,
): Record<string, NodeConfig> {
  const current = configs[id];
  if (!current) return configs;
  const next = { ...current, ...patch } as NodeConfig;
  const nameChanged = current.name !== next.name;
  let result: Record<string, NodeConfig> = { ...configs, [id]: next };
  result = syncSubcategoryConfigsForCategoryUpdate(
    current,
    next,
    result,
    current.name,
    next.name,
    nameChanged,
  );
  if (nameChanged)
    result = applyRenameToConfigs(result, current.name, next.name);
  if (
    current.kind === "model_provider" &&
    next.kind === "model_provider" &&
    (current.is_local === true) !== (next.is_local === true)
  ) {
    for (const [cid, cfg] of Object.entries(result)) {
      if (cfg.kind === "model_config" && cfg.provider === next.name) {
        // biome-ignore lint/style/useNamingConvention: api schema
        result = {
          ...result,
          [cid]: { ...cfg, model: "", gguf_variant: undefined },
        };
      }
    }
  }
  return result;
}

export function isColumnConfig(config: NodeConfig): boolean {
  return COLUMN_KINDS.has(config.kind);
}

export function buildColumn(
  config: NodeConfig,
  configs: Record<string, NodeConfig>,
): RecipeColumn | null {
  const errors: string[] = [];
  if (config.kind === "sampler") return buildSamplerColumn(config, errors);
  if (config.kind === "llm") {
    const column = buildLlmColumn(config, errors);
    return JSON.parse(JSON.stringify(column)) as RecipeColumn;
  }
  if (config.kind === "expression")
    return buildExpressionColumn(config, errors);
  if (config.kind === "validator") {
    const byName = new Map(Object.values(configs).map((c) => [c.name, c]));
    return buildValidatorColumn(config, errors, byName);
  }
  return null;
}

function asNodes(configs: Record<string, NodeConfig>): RecipeNode[] {
  return Object.values(configs).map(
    (config) =>
      ({
        id: config.id,
        type: "builder",
        position: { x: 0, y: 0 },
        data: {},
      }) as unknown as RecipeNode,
  );
}

export interface RecipeEdit {
  /** column nodes whose params (and label) change */
  nodes: FlowNode[];
  group: FlowGroup | null;
}

/** What an edit made in Recipe Studio configs changes in the flow: only touched columns are rebuilt. */
export function commitStudioEdit(
  group: FlowGroup,
  columns: FlowNode[],
  before: StudioView,
  configs: Record<string, NodeConfig>,
  processors: RecipeProcessorConfig[] = before.processors,
): RecipeEdit {
  const nodes: FlowNode[] = [];
  for (const node of columns) {
    const id = before.byNode[node.id];
    const config = id ? configs[id] : undefined;
    if (!config || config === before.configs[id]) continue;
    const column = buildColumn(config, configs);
    if (!column) continue;
    nodes.push({
      ...node,
      label: config.name.slice(0, 200),
      params: { ...node.params, column },
    });
  }
  const recipeLevel =
    Object.values(configs).some(
      (config) =>
        !isColumnConfig(config) && config !== before.configs[config.id],
    ) ||
    Object.values(before.configs).some(
      (config) => !isColumnConfig(config) && !configs[config.id],
    );
  if (!recipeLevel && processors === before.processors) {
    return { nodes, group: null };
  }
  const built = buildRecipePayload(configs, asNodes(configs), [], processors);
  const { columns: _columns, ...recipe } = built.payload.recipe as Record<
    string,
    unknown
  >;
  const config: Record<string, unknown> = { ...(group.config ?? {}) };
  for (const [key, value] of Object.entries(recipe)) {
    if (value === undefined) delete config[key];
    else config[key] = value;
  }
  const ui: Record<string, unknown> = { ...(group.ui ?? {}) };
  for (const [key, value] of Object.entries(built.payload.ui)) {
    if (!LAYOUT_UI.has(key)) ui[key] = value;
  }
  // A tool profile's fetched tool list lives on its Recipe Studio canvas node.
  const toolNodes = (built.payload.ui.nodes ?? []).filter(
    (node) => (node as { node_type?: string }).node_type === "tool_config",
  );
  if (toolNodes.length > 0) ui.nodes = toolNodes;
  else delete ui.nodes;
  return { nodes, group: { ...group, config, ui } };
}

/** A new column node for a config made by a Recipe Studio block factory. */
export function columnNode(
  config: NodeConfig,
  configs: Record<string, NodeConfig>,
  groupId: string,
  pos: [number, number],
  id: string,
): FlowNode | null {
  const column = buildColumn(config, configs);
  if (!column) return null;
  return {
    id,
    type: COLUMN_TYPE,
    v: 1,
    params: { column },
    pos,
    label: config.name,
    group: groupId,
  };
}
