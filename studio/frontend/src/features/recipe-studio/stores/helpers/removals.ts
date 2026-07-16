// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { Edge } from "@xyflow/react";
import type { NodeConfig } from "../../types";
import { isCategoryConfig, isSubcategoryConfig } from "../../utils";
import { applyRemovalToConfig, applyRemovalToConfigs } from "../recipe-studio-helpers";

export function applyNodeRemovals(
  input: { edges: Edge[]; configs: Record<string, NodeConfig> },
  removedIds: string[],
): { edges: Edge[]; configs: Record<string, NodeConfig> } {
  if (removedIds.length === 0) {
    return input;
  }

  const edges = input.edges.filter(
    (edge) => !(removedIds.includes(edge.source) || removedIds.includes(edge.target)),
  );
  let configs: Record<string, NodeConfig> = { ...input.configs };
  const removedNames: string[] = [];

  for (const id of removedIds) {
    const removed = configs[id];
    delete configs[id];
    if (removed?.name) {
      removedNames.push(removed.name);
    }

    if (isCategoryConfig(removed)) {
      const removedName = removed.name;
      for (const config of Object.values(configs)) {
        if (!isSubcategoryConfig(config)) {
          continue;
        }
        if (config.subcategory_parent !== removedName) {
          continue;
        }
        configs[config.id] = {
          ...config,
          // biome-ignore lint/style/useNamingConvention: api schema
          subcategory_parent: "",
          // biome-ignore lint/style/useNamingConvention: api schema
          subcategory_mapping: {},
        };
      }
    }
  }

  for (const name of removedNames) {
    configs = applyRemovalToConfigs(configs, name);
  }

  return { edges, configs };
}

export function applyEdgeRemovals(
  configs: Record<string, NodeConfig>,
  removedEdges: Edge[],
): Record<string, NodeConfig> {
  if (removedEdges.length === 0) {
    return configs;
  }

  let next = configs;
  for (const edge of removedEdges) {
    const source = next[edge.source];
    const target = next[edge.target];
    if (!(source && target)) {
      continue;
    }
    const updated = applyRemovalToConfig(target, source.name);
    if (updated !== target) {
      if (next === configs) {
        next = { ...configs };
      }
      next[target.id] = updated;
    }
    if (
      source.kind === "validator" &&
      target.kind === "llm" &&
      target.llm_type === "code"
    ) {
      const sourceUpdated = applyRemovalToConfig(source, target.name);
      if (sourceUpdated !== source) {
        if (next === configs) {
          next = { ...configs };
        }
        next[source.id] = sourceUpdated;
      }
    }
  }
  return next;
}
