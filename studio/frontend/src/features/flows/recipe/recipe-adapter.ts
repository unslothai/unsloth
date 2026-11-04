// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A Recipe group seen through Recipe Studio's own configs, so its block dialogs, reference sync and payload
// builders edit a flow without the Recipe Studio store. Loaded with the editor dialogs, not with the canvas.

import { syncSubcategoryConfigsForCategoryUpdate } from "@/features/recipe-studio/stores/helpers/edge-sync";
import { applyRenameToConfigs } from "@/features/recipe-studio/stores/helpers/reference-sync";
import { renameRoot, replaceRef } from "@/features/recipe-studio/utils/refs";
import type {
  NodeConfig,
  RecipeNode,
  RecipeProcessorConfig,
  ToolProfileConfig,
} from "@/features/recipe-studio/types";
import { makeToolProfileConfig } from "@/features/recipe-studio/utils/config-factories";
import { importRecipePayload } from "@/features/recipe-studio/utils/import/importer";
import { buildRecipePayload } from "@/features/recipe-studio/utils/payload/build-payload";
import { buildToolProfilePayload } from "@/features/recipe-studio/utils/payload/builders-llm";
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
  columnOf,
  groupPayload,
  pendingToolProfiles,
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

/** Recipe Studio's own pre-server checks (a used provider with no endpoint, a preset with no model, ...). */
export function clientErrors(group: FlowGroup, columns: FlowNode[]): string[] {
  const view = studioView(group, columns);
  if (view.errors.length) return view.errors;
  return buildRecipePayload(
    view.configs,
    asNodes(view.configs),
    [],
    view.processors,
  ).errors;
}

/** The group as Recipe Studio configs, the same ones its importer makes for a saved recipe. */
export function studioView(group: FlowGroup, columns: FlowNode[]): StudioView {
  // As Recipe Studio loads a saved recipe: a loaded document seed keeps its files.
  const result = importRecipePayload(
    JSON.stringify(groupPayload(group, columns)),
    { preserveUnstructuredUploads: true },
  );
  if (!result.snapshot) {
    return { configs: {}, byNode: {}, processors: [], errors: result.errors };
  }
  const configs = { ...result.snapshot.configs };
  // A tool profile with no server yet lives only in the ui nodes (Recipe Studio's payload drops it).
  for (const name of pendingToolProfiles(group)) {
    if (Object.values(configs).some((c) => c.name === name)) continue;
    const id = `flows-tools-${name}`;
    configs[id] = { ...makeToolProfileConfig(id, []), name };
  }
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
/** A tool server is one recipe-wide definition shared by name: an edit made through one tool profile is
 * made in every profile that names it, so the copies never disagree (the payload keeps only one). */
function withSharedMcpProviders(
  configs: Record<string, NodeConfig>,
  before: ToolProfileConfig,
  after: ToolProfileConfig,
): Record<string, NodeConfig> {
  const was = new Map(before.mcp_providers.map((p) => [p.id, p]));
  const edited = after.mcp_providers.filter((p) => {
    const old = was.get(p.id);
    return old && old.name === p.name && JSON.stringify(old) !== JSON.stringify(p);
  });
  if (!edited.length) return configs;
  const byName = new Map(edited.map((p) => [p.name, p]));
  let result = configs;
  for (const [cid, cfg] of Object.entries(configs)) {
    if (cid === after.id || cfg.kind !== "tool_config") continue;
    if (!cfg.mcp_providers.some((p) => byName.has(p.name))) continue;
    // biome-ignore lint/style/useNamingConvention: ui schema
    const mcp_providers = cfg.mcp_providers.map((p) => {
      const shared = byName.get(p.name);
      return shared ? { ...shared, id: p.id } : p;
    });
    result = { ...result, [cid]: { ...cfg, mcp_providers } };
  }
  return result;
}

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
  if (next.kind === "tool_config" && current.kind === "tool_config")
    result = withSharedMcpProviders(result, current, next);
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

/** The schema transform's templates name columns too: a column renamed in this edit is renamed
 * there, so the transform still reads it. Unchanged processors keep their identity. */
function columnRenames(
  before: StudioView,
  configs: Record<string, NodeConfig>,
): [string, string][] {
  return Object.entries(before.configs).flatMap(([id, was]): [string, string][] => {
    const now = configs[id];
    if (!now || !isColumnConfig(was) || !was.name || !now.name || was.name === now.name) return [];
    return [[was.name, now.name]];
  });
}

/** A schema transform's template (JSON text) with a column renamed in its Jinja strings: read as
 * JSON first, so a quoted key in an expression (`record["topic"]`) is not escaped out of reach. */
function renamedTemplate(template: string, from: string, to: string): string {
  let parsed: unknown;
  try {
    parsed = JSON.parse(template);
  } catch {
    return replaceRef(template, from, to);
  }
  let changed = false;
  const walk = (value: unknown): unknown => {
    if (typeof value === "string") {
      const next = replaceRef(value, from, to);
      if (next !== value) changed = true;
      return next;
    }
    if (Array.isArray(value)) return value.map(walk);
    if (value && typeof value === "object")
      return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, walk(v)]));
    return value;
  };
  const next = walk(parsed);
  return changed ? JSON.stringify(next, null, 2) : template;
}

function renamedProcessors(
  before: StudioView,
  configs: Record<string, NodeConfig>,
): RecipeProcessorConfig[] {
  let processors = before.processors;
  for (const [from, to] of columnRenames(before, configs)) {
    const next = processors.map((processor) => {
      const template = renamedTemplate(processor.template, from, to);
      return template === processor.template ? processor : { ...processor, template };
    });
    if (next.some((processor, i) => processor !== processors[i])) processors = next;
  }
  return processors;
}

/** The group's processors the forms do not show (a named drop_columns), with renamed columns
 * renamed in their column lists; null when none changed. */
function renamedDrops(
  group: FlowGroup,
  renames: [string, string][],
): unknown[] | null {
  const raw = Array.isArray(group.config?.processors) ? group.config.processors : [];
  if (!renames.length) return null;
  const map = new Map(renames);
  let changed = false;
  const next = raw.map((p) => {
    const item = p as Record<string, unknown> | null;
    if (!item || item.processor_type !== "drop_columns" || !Array.isArray(item.column_names))
      return p;
    const names = item.column_names.map((n) => (typeof n === "string" ? (map.get(n) ?? n) : n));
    if (names.every((n, i) => n === (item.column_names as unknown[])[i])) return p;
    changed = true;
    // biome-ignore lint/style/useNamingConvention: api schema
    return { ...item, column_names: names };
  });
  return changed ? next : null;
}

/** The field naming each entry of a recipe-level list, as the forms name its block. */
const ENTRY_NAME: Record<string, string> = {
  // biome-ignore lint/style/useNamingConvention: api schema
  model_providers: "name",
  // biome-ignore lint/style/useNamingConvention: api schema
  mcp_providers: "name",
  // biome-ignore lint/style/useNamingConvention: api schema
  model_configs: "alias",
  // biome-ignore lint/style/useNamingConvention: api schema
  tool_configs: "tool_alias",
};

/** Fields the forms edit (builders-model.ts): an edited entry takes these from the form, set or
 * left out; every other saved field stays. Lists not named here are taken from the form whole. */
const FORM_FIELDS: Record<string, string[]> = {
  // biome-ignore lint/style/useNamingConvention: api schema
  model_providers: ["name", "endpoint", "provider_type", "api_key_env", "api_key", "extra_headers", "extra_body", "is_local"],
  // biome-ignore lint/style/useNamingConvention: api schema
  model_configs: ["alias", "model", "gguf_variant", "provider", "skip_health_check", "inference_parameters"],
};
const FORM_INFERENCE = ["temperature", "top_p", "max_tokens", "timeout", "extra_body"];

function without(record: Record<string, unknown>, keys: string[]): Record<string, unknown> {
  return Object.fromEntries(Object.entries(record).filter(([k]) => !keys.includes(k)));
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);

/** `form` with each field the form left as it was (`prior` = the form's value before the edit) taken from
 * `original`, so a value the form cannot show (a uniform temperature) survives an edit of another field. */
function untouchedFromSaved(
  form: Record<string, unknown>,
  prior: Record<string, unknown> | null,
  original: Record<string, unknown>,
  fields: string[],
): Record<string, unknown> {
  const next = { ...form };
  for (const f of fields) {
    if (!prior || !same(prior[f], form[f])) continue;
    if (f in original) next[f] = original[f];
    else delete next[f];
  }
  return next;
}

/** `built` with every part the edit left as it was (`prior` = the same thing built before the edit)
 * taken from `saved`, nested objects part by part. */
function keepUntouched(built: unknown, saved: unknown, prior: unknown): unknown {
  if (saved === undefined) return built;
  if (same(prior, built)) return saved;
  const b = asRecord(built);
  const v = asRecord(saved);
  const p = asRecord(prior);
  if (!b || !v || !p) return built;
  const next: Record<string, unknown> = {};
  for (const k of new Set([...Object.keys(b), ...Object.keys(v), ...Object.keys(p)])) {
    const kept = same(p[k], b[k]) ? v[k] : keepUntouched(b[k], v[k], p[k]);
    if (kept !== undefined) next[k] = kept;
  }
  return next;
}

/** A provider, preset or tool profile the edit did not touch is kept exactly as saved, settings the
 * forms do not show included (a preset's max_parallel_requests, a uniform temperature); an edited
 * or renamed one keeps those settings and every field its form left alone, and takes what the form changed. */
function keepUnshown(
  key: string,
  built: unknown,
  saved: unknown,
  before: StudioView,
  configs: Record<string, NodeConfig>,
  priorBuilt: unknown,
): unknown {
  const field = ENTRY_NAME[key];
  if (!field || !Array.isArray(built) || !Array.isArray(saved)) return built;
  const byName = new Map(
    saved.flatMap((entry) => {
      const name = asRecord(entry)?.[field];
      return typeof name === "string" ? [[name, entry as Record<string, unknown>]] : [];
    }),
  );
  return built.map((entry) => {
    const item = asRecord(entry);
    const name = item?.[field];
    if (!item || typeof name !== "string") return entry;
    // The block this entry came from, found by its id so a rename still finds what was saved.
    const config = Object.values(configs).find((c) => !isColumnConfig(c) && c.name === name);
    const was = config ? before.configs[config.id] : undefined;
    const original = byName.get(was?.name ?? name);
    if (!original) return entry;
    if (config && was === config) return original;
    const fields = FORM_FIELDS[key];
    if (!fields) return entry;
    const prior = Array.isArray(priorBuilt)
      ? (priorBuilt.map(asRecord).find((e) => e?.[field] === (was?.name ?? name)) ?? null)
      : null;
    const shown = fields.filter((f) => f !== "inference_parameters");
    const merged: Record<string, unknown> = {
      ...without(original, fields),
      ...untouchedFromSaved(without(item, ["inference_parameters"]), prior, original, shown),
    };
    if ("inference_parameters" in item) merged.inference_parameters = item.inference_parameters;
    const savedParams = asRecord(original.inference_parameters);
    if (key === "model_configs" && savedParams) {
      // biome-ignore lint/style/useNamingConvention: api schema
      merged.inference_parameters = {
        ...without(savedParams, FORM_INFERENCE),
        ...untouchedFromSaved(
          asRecord(item.inference_parameters) ?? {},
          prior ? (asRecord(prior.inference_parameters) ?? {}) : null,
          savedParams,
          FORM_INFERENCE,
        ),
      };
    }
    return merged;
  });
}

/** The view an open form compares its next edit against once `configs` is saved: the saved configs,
 * with their names and renamed processors, so a later edit is told apart from this one. */
export function savedView(before: StudioView, configs: Record<string, NodeConfig>): StudioView {
  return { ...before, configs, processors: renamedProcessors(before, configs) };
}

const PARENT_FIELDS = ["category", "reference_column_name", "subcategory_parent"];

/** A sampler's conditions (`tier == 'premium'`) with renamed columns renamed; the same object when none is. */
function renamedConditions(conditions: unknown, renames: Map<string, string>): unknown {
  const record = asRecord(conditions);
  if (!record || !renames.size) return conditions;
  let changed = false;
  const next = Object.fromEntries(
    Object.entries(record).map(([rule, value]) => {
      let renamed = rule;
      for (const [from, to] of renames) renamed = renameRoot(renamed, from, to);
      if (renamed !== rule) changed = true;
      // An override names its parent column outright too (a subcategory's `category`).
      const params = asRecord(value);
      let override = value;
      for (const field of PARENT_FIELDS) {
        const to = typeof params?.[field] === "string" ? renames.get(params[field] as string) : undefined;
        if (to === undefined) continue;
        override = { ...(asRecord(override) ?? {}), [field]: to };
        changed = true;
      }
      return [renamed, override];
    }),
  );
  return changed ? next : conditions;
}

/** An AI column's image contexts with renamed columns renamed; the same list when none is. */
function renamedImages(list: unknown, renames: Map<string, string>): unknown {
  if (!Array.isArray(list) || !renames.size) return list;
  let changed = false;
  const next = list.map((entry) => {
    const name = (entry as Record<string, unknown> | null)?.column_name;
    const to = typeof name === "string" ? renames.get(name) : undefined;
    if (to === undefined) return entry;
    changed = true;
    // biome-ignore lint/style/useNamingConvention: api schema
    return { ...(entry as Record<string, unknown>), column_name: to };
  });
  return changed ? next : list;
}

/** What an edit made in Recipe Studio configs changes in the flow: only touched columns are rebuilt. */
export function commitStudioEdit(
  group: FlowGroup,
  columns: FlowNode[],
  before: StudioView,
  configs: Record<string, NodeConfig>,
  processors: RecipeProcessorConfig[] = renamedProcessors(before, configs),
): RecipeEdit {
  const nodes: FlowNode[] = [];
  const renames = new Map(columnRenames(before, configs));
  for (const node of columns) {
    const id = before.byNode[node.id];
    const config = id ? configs[id] : undefined;
    const saved = columnOf(node);
    const images = renamedImages(saved.multi_modal_context, renames);
    const conditions = renamedConditions(saved.conditional_params, renames);
    if (!config) continue;
    if (config === before.configs[id]) {
      // What the forms do not show (a second image context, a numeric sampler's conditions) is
      // renamed in place.
      if (images !== saved.multi_modal_context || conditions !== saved.conditional_params)
        nodes.push({
          ...node,
          params: {
            ...node.params,
            column: {
              ...saved,
              ...(images === undefined ? {} : { multi_modal_context: images }),
              ...(conditions === undefined ? {} : { conditional_params: conditions }),
            },
          },
        });
      continue;
    }
    const built = buildColumn(config, configs);
    if (!built) continue;
    const prior = buildColumn(before.configs[id], before.configs);
    const column = prior
      ? (untouchedFromSaved(
          built as Record<string, unknown>,
          prior as Record<string, unknown>,
          saved,
          [...new Set([...Object.keys(prior), ...Object.keys(built), ...Object.keys(saved)])],
        ) as typeof built)
      : built;
    const kept = column as Record<string, unknown>;
    // A form that changed one setting of a sampler of the same type keeps the settings it does not
    // show (a uniform's decimal_places, a uuid's short_form).
    if (
      prior &&
      same(prior.column_type, built.column_type) &&
      same((prior as Record<string, unknown>).sampler_type, (built as Record<string, unknown>).sampler_type)
    )
      kept.params = keepUntouched(kept.params, saved.params, (prior as Record<string, unknown>).params);
    if (kept.conditional_params === saved.conditional_params && conditions !== undefined)
      kept.conditional_params = conditions;
    // Only a rename changed the form's image context: the saved list keeps its other images and
    // their attributes, renamed where they name the column.
    const formImages = (c: unknown) => (c as Record<string, unknown> | null)?.multi_modal_context;
    if (
      Array.isArray(saved.multi_modal_context) &&
      JSON.stringify(renamedImages(formImages(prior), renames) ?? null) ===
        JSON.stringify(formImages(built) ?? null)
    )
      (column as Record<string, unknown>).multi_modal_context = images;
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
  const drops = renamedDrops(group, columnRenames(before, configs));
  if (!recipeLevel && processors === before.processors && !drops) {
    return { nodes, group: null };
  }
  const built = buildRecipePayload(configs, asNodes(configs), [], processors);
  const { columns: _columns, ...recipe } = built.payload.recipe as Record<
    string,
    unknown
  >;
  const priorRecipe = buildRecipePayload(
    before.configs,
    asNodes(before.configs),
    [],
    before.processors,
  ).payload.recipe as Record<string, unknown>;
  const config: Record<string, unknown> = { ...(group.config ?? {}) };
  for (const [key, value] of Object.entries(recipe)) {
    const saved = group.config?.[key];
    if (!ENTRY_NAME[key] && key !== "processors") {
      // The seed and other single settings: what the edit left alone stays as saved (an HF seed's
      // split and subset, which the form does not hold), what it changed comes from the form.
      const kept = keepUntouched(value, saved, priorRecipe[key]);
      if (kept === undefined) delete config[key];
      else config[key] = kept;
    } else if (value === undefined) delete config[key];
    else
      config[key] = keepUnshown(
        key,
        value,
        saved,
        before,
        configs,
        priorRecipe[key],
      );
  }
  // Recipe Studio's forms carry schema transforms and the seed's drop list only; any other
  // processor (a named drop_columns) is kept as it was, not lost to an unrelated edit.
  const others = (
    drops ?? (Array.isArray(group.config?.processors) ? group.config.processors : [])
  ).filter((p): p is Record<string, unknown> => {
    if (!p || typeof p !== "object") return false;
    const { processor_type: type, name } = p as Record<string, unknown>;
    return type !== "schema_transform" && !(type === "drop_columns" && name === "drop_seed_columns");
  });
  if (others.length > 0)
    config.processors = [
      ...(Array.isArray(config.processors) ? config.processors : []),
      ...others,
    ];
  keepIncompleteTools(config, group.config ?? {}, configs);
  const seed = Object.values(configs).find((c) => c.kind === "seed");
  if (seed?.kind === "seed" && config.seed_config === undefined) {
    // A seed whose form is mid-edit (an End cleared to retype) builds nothing: keep its last
    // valid config, so the card and its form stay; only removing the seed block deletes it.
    const kept = group.config?.seed_config as
      | { source?: { seed_type?: string } }
      | undefined;
    const github = seed.seed_source_type === "github_repo";
    if (kept !== undefined && (!github || kept.source?.seed_type === "github_repo"))
      config.seed_config = kept;
    else if (github)
      // A GitHub seed with no repository yet builds nothing; keep a placeholder so its form opens.
      // biome-ignore lint/style/useNamingConvention: api schema
      config.seed_config = { source: { seed_type: "github_repo", repos: [] } };
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

/** A tool profile whose form is mid-edit (a server's endpoint cleared to retype) builds without
 * that server, or not at all: it keeps its last saved profile and servers, so the card and its
 * settings stay. Only removing the profile or a server deletes them. */
function keepIncompleteTools(
  config: Record<string, unknown>,
  saved: Record<string, unknown>,
  configs: Record<string, NodeConfig>,
): void {
  const list = (value: unknown) =>
    (Array.isArray(value) ? value : []) as Record<string, unknown>[];
  for (const tool of Object.values(configs)) {
    if (tool.kind !== "tool_config") continue;
    // A profile with no servers left had them removed on purpose: nothing of it is kept.
    const present = new Set(tool.mcp_providers.map((p) => p.name.trim()));
    if (!present.size) continue;
    const errors: string[] = [];
    buildToolProfilePayload(tool, errors);
    if (!errors.length) continue;
    const kept = list(saved.tool_configs).find((t) => t.tool_alias === tool.name);
    if (!kept) continue;
    const tools = list(config.tool_configs).filter((t) => t.tool_alias !== tool.name);
    const providers = Array.isArray(kept.providers)
      ? (kept.providers as unknown[]).filter((name) => present.has(String(name)))
      : kept.providers;
    // biome-ignore lint/style/useNamingConvention: api schema
    config.tool_configs = [...tools, { ...kept, providers }];
    const names = list(config.mcp_providers).map((p) => p.name);
    const uses = Array.isArray(kept.providers) ? (kept.providers as unknown[]) : [];
    // Only servers still on the profile come back; one removed from it stays removed.
    const servers = list(saved.mcp_providers).filter(
      (p) =>
        uses.includes(p.name) &&
        present.has(String(p.name)) &&
        !names.includes(p.name),
    );
    // biome-ignore lint/style/useNamingConvention: api schema
    if (servers.length) config.mcp_providers = [...list(config.mcp_providers), ...servers];
  }
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
