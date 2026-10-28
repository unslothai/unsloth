// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A recipe group's seed, provider connections, model presets and tool profiles as canvas cards. They are not
// flow nodes: the group's recipe config is their source, and a dragged card keeps its spot in the group's `ui`.

import type { DocGraph } from "../stores/doc-store";
import { GROUP_KEY } from "../stores/doc-store";
import type { FlowGroup } from "../types";
import {
  columnOf,
  groupColumns,
  groupGenerate,
  isRecipeGroup,
} from "./recipe-columns";
import { SEED_BLOCKS, SEED_TYPE_BLOCK } from "./recipe-kinds";

export const SETUP_PREFIX = "recipe-setup:";
export const SETUP_POSITIONS = "flows_setup_positions";
const GAP = 24;
const CARD_HEIGHT = 96;

export type SetupKind = "seed" | "model_provider" | "model_config" | "tool_config";

export interface SetupCard {
  /** canvas node id: `recipe-setup:<group>:<kind>:<name>` */
  id: string;
  groupId: string;
  kind: SetupKind;
  /** the config's name (alias for models and tools); "seed" for the seed */
  name: string;
  title: string;
  subtitle: string;
  fields: { label: string; value: string }[];
  /** what still has to be done before it runs */
  warning?: string;
}

export interface SetupLink {
  id: string;
  source: string;
  target: string;
}

export interface SetupElement {
  card: SetupCard;
  x: number;
  y: number;
}

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function list(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value) ? value.map(record) : [];
}

function text(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

export function setupId(groupId: string, kind: SetupKind, name: string): string {
  return `${SETUP_PREFIX}${groupId}:${kind}:${name}`;
}

export function isSetupId(id: string): boolean {
  return id.startsWith(SETUP_PREFIX);
}

/** The group, kind and name inside a card id. */
export function parseSetupId(
  id: string,
): { groupId: string; kind: SetupKind; name: string } | null {
  if (!isSetupId(id)) return null;
  const [groupId, kind, ...rest] = id.slice(SETUP_PREFIX.length).split(":");
  return { groupId, kind: kind as SetupKind, name: rest.join(":") };
}

function seedCard(group: FlowGroup): SetupCard | null {
  const seed = record(group.config?.seed_config);
  const source = record(seed.source);
  const type = text(source.seed_type);
  if (!type) return null;
  const block = SEED_BLOCKS.find((b) => b.type === SEED_TYPE_BLOCK[type]);
  const ui = record(group.ui);
  const fields: { label: string; value: string }[] = [];
  let warning: string | undefined;
  if (type === "hf") {
    const path = text(source.path);
    const repo = path.replace(/^datasets\//, "").split("/").slice(0, 2).join("/");
    fields.push({ label: "Dataset", value: repo || "not set" });
  } else if (type === "github_repo") {
    const repos = Array.isArray(source.repos) ? source.repos.map(String) : [];
    fields.push({ label: "Repos", value: repos.join(", ") || "not set" });
  } else {
    const names = Array.isArray(ui.unstructured_file_names)
      ? ui.unstructured_file_names.map(String)
      : [];
    const file = text(ui.local_file_name) || names.join(", ");
    const hasPath =
      !!text(source.path) ||
      (Array.isArray(source.paths) && source.paths.some(Boolean));
    fields.push({ label: "File", value: file || "not loaded" });
    if (!hasPath) warning = "Load the seed first";
  }
  const columns = Array.isArray(ui.seed_columns) ? ui.seed_columns : [];
  if (columns.length)
    fields.push({ label: "Columns", value: columns.map(String).join(", ") });
  return {
    id: setupId(group.id, "seed", "seed"),
    groupId: group.id,
    kind: "seed",
    name: "seed",
    title: block?.label ?? "Seed data",
    subtitle: "Seed data",
    fields,
    warning,
  };
}

/** A group's setup as cards, wired to what uses them. */
export function recipeSetup(
  graph: DocGraph,
  group: FlowGroup,
): { cards: SetupCard[]; links: SetupLink[] } {
  const config = group.config ?? {};
  const cards: SetupCard[] = [];
  const links: SetupLink[] = [];
  const seed = seedCard(group);
  if (seed) cards.push(seed);
  const providers = new Map<string, { id: string; local: boolean }>();
  for (const provider of list(config.model_providers)) {
    const name = text(provider.name);
    if (!name) continue;
    const local = provider.is_local === true;
    const id = setupId(group.id, "model_provider", name);
    providers.set(name, { id, local });
    cards.push({
      id,
      groupId: group.id,
      kind: "model_provider",
      name,
      title: name,
      subtitle: "Provider connection",
      fields: local
        ? [{ label: "Uses", value: "Unsloth Studio's chat model" }]
        : [
            { label: "Endpoint", value: text(provider.endpoint) || "not set" },
          ],
    });
  }
  const models = new Map<string, string>();
  for (const model of list(config.model_configs)) {
    const alias = text(model.alias);
    if (!alias) continue;
    const id = setupId(group.id, "model_config", alias);
    models.set(alias, id);
    const provider = providers.get(text(model.provider));
    const inference = record(model.inference_parameters);
    const name = text(model.model);
    const fields = [
      {
        label: "Model",
        value: name || (provider?.local ? "The loaded chat model" : "not set"),
      },
    ];
    if (inference.temperature !== undefined)
      fields.push({ label: "Temp", value: String(inference.temperature) });
    cards.push({
      id,
      groupId: group.id,
      kind: "model_config",
      name: alias,
      title: alias,
      subtitle: "Model preset",
      fields,
    });
    if (provider)
      links.push({ id: `${provider.id}>${id}`, source: provider.id, target: id });
  }
  const tools = new Map<string, string>();
  for (const tool of list(config.tool_configs)) {
    const alias = text(tool.tool_alias);
    if (!alias) continue;
    const id = setupId(group.id, "tool_config", alias);
    tools.set(alias, id);
    const servers = Array.isArray(tool.providers) ? tool.providers.map(String) : [];
    cards.push({
      id,
      groupId: group.id,
      kind: "tool_config",
      name: alias,
      title: alias,
      subtitle: "Tool access",
      fields: [{ label: "Servers", value: servers.join(", ") || "none" }],
    });
  }
  for (const node of groupColumns(graph, group.id)) {
    const column = columnOf(node);
    const model = models.get(text(column.model_alias));
    if (model) links.push({ id: `${model}>${node.id}`, source: model, target: node.id });
    const tool = tools.get(text(column.tool_alias));
    if (tool) links.push({ id: `${tool}>${node.id}`, source: tool, target: node.id });
  }
  return { cards, links };
}

type Size = { width: number; height: number };
type Point = { x: number; y: number };

function savedPositions(group: FlowGroup): Record<string, [number, number]> {
  const saved = group.ui?.[SETUP_POSITIONS];
  return saved && typeof saved === "object"
    ? (saved as Record<string, [number, number]>)
    : {};
}

/** Where each card sits: its saved spot in the group, else stacked under the group's Recipe Dataset step. */
export function setupLayout(
  graph: DocGraph,
  drag: Record<string, Point> | null,
  sizes: ReadonlyMap<string, Size>,
): {
  elements: SetupElement[];
  links: SetupLink[];
  /** recipe groups whose frame grows to hold their cards: group id -> height */
  heights: Record<string, number>;
} {
  const elements: SetupElement[] = [];
  const links: SetupLink[] = [];
  const heights: Record<string, number> = {};
  for (const groupId of graph.groupOrder) {
    const group = graph.groups[groupId];
    if (!isRecipeGroup(group)) continue;
    const { cards, links: groupLinks } = recipeSetup(graph, group);
    if (!cards.length) continue;
    links.push(...groupLinks);
    const origin = drag?.[GROUP_KEY + groupId] ?? {
      x: group.rect[0],
      y: group.rect[1],
    };
    const saved = savedPositions(group);
    const generate = groupGenerate(graph, groupId);
    const anchor = generate
      ? (drag?.[generate.id] ?? { x: generate.pos[0], y: generate.pos[1] })
      : { x: origin.x + 40, y: origin.y + 60 };
    let nextY =
      anchor.y +
      (generate ? (sizes.get(generate.id)?.height ?? 140) : 0) +
      GAP;
    let bottom = 0;
    for (const card of cards) {
      let pos = drag?.[card.id];
      if (!pos) {
        const rel = saved[card.id.slice(SETUP_PREFIX.length)];
        if (Array.isArray(rel))
          pos = { x: origin.x + rel[0], y: origin.y + rel[1] };
      }
      if (!pos) {
        pos = { x: anchor.x, y: nextY };
        nextY += (sizes.get(card.id)?.height ?? CARD_HEIGHT) + GAP / 2;
      }
      elements.push({ card, x: pos.x, y: pos.y });
      bottom = Math.max(
        bottom,
        pos.y + (sizes.get(card.id)?.height ?? CARD_HEIGHT),
      );
    }
    const needed = bottom - origin.y + GAP;
    if (needed > group.rect[3]) heights[groupId] = Math.round(needed);
  }
  return { elements, links, heights };
}

/** The group with a dragged card's spot saved relative to its frame. */
export function withSetupPosition(
  group: FlowGroup,
  cardId: string,
  pos: [number, number],
): FlowGroup {
  return {
    ...group,
    ui: {
      ...(group.ui ?? {}),
      [SETUP_POSITIONS]: {
        ...savedPositions(group),
        [cardId.slice(SETUP_PREFIX.length)]: [
          Math.round(pos[0] - group.rect[0]),
          Math.round(pos[1] - group.rect[1]),
        ],
      },
    },
  };
}
