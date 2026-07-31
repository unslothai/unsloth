// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type Edge, addEdge } from "@xyflow/react";
import type {
  LayoutDirection,
  ModelConfig,
  NodeConfig,
  SamplerConfig,
  ValidatorConfig,
} from "../../types";
import { applyRecipeConnection } from "../../utils/graph";
import { isCategoryConfig, isSubcategoryConfig } from "../../utils";
import { HANDLE_IDS } from "../../utils/handles";

function findNodeIdByName(
  configs: Record<string, NodeConfig>,
  name: string,
): string | null {
  const entry = Object.entries(configs).find(
    ([, config]) => config.name === name,
  );
  return entry ? entry[0] : null;
}

function addRecipeEdge(edges: Edge[], source: string, target: string): Edge[] {
  return addEdge(
    {
      source,
      target,
      sourceHandle: HANDLE_IDS.dataOut,
      targetHandle: HANDLE_IDS.dataIn,
      type: "canvas",
    },
    edges,
  );
}

function addValidatorSemanticEdge(
  edges: Edge[],
  source: string,
  target: string,
): Edge[] {
  return addEdge(
    {
      source,
      target,
      sourceHandle: HANDLE_IDS.dataOut,
      targetHandle: HANDLE_IDS.dataIn,
      type: "semantic",
    },
    edges,
  );
}

function removeTargetEdges(edges: Edge[], targetId: string): Edge[] {
  return edges.filter((edge) => edge.target !== targetId);
}

function removeTargetEdgesBySource(
  edges: Edge[],
  configs: Record<string, NodeConfig>,
  targetId: string,
  shouldRemove: (source: NodeConfig | undefined) => boolean,
): Edge[] {
  return edges.filter((edge) => {
    if (edge.target !== targetId) {
      return true;
    }
    return !shouldRemove(configs[edge.source]);
  });
}

export function syncEdgesForConfigPatch(
  current: NodeConfig,
  patch: Partial<NodeConfig>,
  configs: Record<string, NodeConfig>,
  edges: Edge[],
  layoutDirection: LayoutDirection,
): Edge[] {
  let nextEdges = edges;

  const hasParentPatch = Object.prototype.hasOwnProperty.call(
    patch,
    "subcategory_parent",
  );
  if (isSubcategoryConfig(current) && hasParentPatch) {
    const nextParent = (patch as Partial<SamplerConfig>).subcategory_parent ?? "";
    const parentId = nextParent ? findNodeIdByName(configs, nextParent) : null;
    nextEdges = removeTargetEdges(nextEdges, current.id);
    if (parentId) {
      nextEdges = addRecipeEdge(nextEdges, parentId, current.id);
    }
  }

  const hasProviderPatch = Object.prototype.hasOwnProperty.call(
    patch,
    "provider",
  );
  if (current.kind === "model_config" && hasProviderPatch) {
    const nextProvider = (patch as Partial<ModelConfig>).provider ?? "";
    if (nextProvider.trim() === current.provider.trim()) {
      return nextEdges;
    }
    nextEdges = removeTargetEdgesBySource(
      nextEdges,
      configs,
      current.id,
      (source) => Boolean(source && source.kind === "model_provider"),
    );
    if (nextProvider) {
      const providerId = findNodeIdByName(configs, nextProvider);
      if (providerId) {
        const result = applyRecipeConnection(
          {
            source: providerId,
            sourceHandle: HANDLE_IDS.semanticOut,
            target: current.id,
            targetHandle: HANDLE_IDS.semanticIn,
          },
          configs,
          nextEdges,
          layoutDirection,
        );
        nextEdges = result.edges;
      }
    }
  }

  const hasReferencePatch = Object.prototype.hasOwnProperty.call(
    patch,
    "reference_column_name",
  );
  if (
    current.kind === "sampler" &&
    current.sampler_type === "timedelta" &&
    hasReferencePatch
  ) {
    const nextReference =
      (patch as Partial<SamplerConfig>).reference_column_name ?? "";
    nextEdges = removeTargetEdgesBySource(
      nextEdges,
      configs,
      current.id,
      (source) =>
        Boolean(
          source &&
            source.kind === "sampler" &&
            source.sampler_type === "datetime",
        ),
    );
    if (nextReference) {
      const referenceId = findNodeIdByName(configs, nextReference);
      const source = referenceId ? configs[referenceId] : null;
      if (
        referenceId &&
        source &&
        source.kind === "sampler" &&
        source.sampler_type === "datetime"
      ) {
        nextEdges = addRecipeEdge(nextEdges, referenceId, current.id);
      }
    }
  }

  const hasModelAliasPatch = Object.prototype.hasOwnProperty.call(
    patch,
    "model_alias",
  );
  if (current.kind === "llm" && hasModelAliasPatch) {
    const nextAlias =
      (patch as Partial<NodeConfig> & { model_alias?: string }).model_alias ?? "";
    if (nextAlias.trim() === current.model_alias.trim()) {
      return nextEdges;
    }
    nextEdges = removeTargetEdgesBySource(
      nextEdges,
      configs,
      current.id,
      (source) => Boolean(source && source.kind === "model_config"),
    );
    if (nextAlias) {
      const modelConfigId = findNodeIdByName(configs, nextAlias);
      if (modelConfigId) {
        const result = applyRecipeConnection(
          {
            source: modelConfigId,
            sourceHandle: HANDLE_IDS.semanticOut,
            target: current.id,
            targetHandle: HANDLE_IDS.semanticIn,
          },
          configs,
          nextEdges,
          layoutDirection,
        );
        nextEdges = result.edges;
      }
    }
  }

  const hasToolAliasPatch = Object.prototype.hasOwnProperty.call(
    patch,
    "tool_alias",
  );
  if (current.kind === "llm" && hasToolAliasPatch) {
    const nextAlias =
      (patch as Partial<NodeConfig> & { tool_alias?: string }).tool_alias ?? "";
    if (nextAlias.trim() === (current.tool_alias ?? "").trim()) {
      return nextEdges;
    }
    nextEdges = removeTargetEdgesBySource(
      nextEdges,
      configs,
      current.id,
      (source) => Boolean(source && source.kind === "tool_config"),
    );
    if (nextAlias) {
      const toolConfigId = findNodeIdByName(configs, nextAlias);
      if (toolConfigId) {
        const result = applyRecipeConnection(
          {
            source: toolConfigId,
            sourceHandle: HANDLE_IDS.semanticOut,
            target: current.id,
            targetHandle: HANDLE_IDS.semanticIn,
          },
          configs,
          nextEdges,
          layoutDirection,
        );
        nextEdges = result.edges;
      }
    }
  }

  const hasValidatorTargetsPatch = Object.prototype.hasOwnProperty.call(
    patch,
    "target_columns",
  );
  if (current.kind === "validator" && hasValidatorTargetsPatch) {
    const nextTargets =
      ((patch as Partial<ValidatorConfig>).target_columns ?? [])
        .map((value) => value.trim())
        .filter(Boolean);
    nextEdges = nextEdges.filter((edge) => {
      if (edge.source !== current.id && edge.target !== current.id) {
        return true;
      }
      const otherId = edge.source === current.id ? edge.target : edge.source;
      const other = configs[otherId];
      return !(
        other &&
        other.kind === "llm" &&
        other.llm_type === "code"
      );
    });
    const nextTargetName = nextTargets[0];
    if (nextTargetName) {
      const targetId = findNodeIdByName(configs, nextTargetName);
      const target = targetId ? configs[targetId] : null;
      if (
        targetId &&
        target &&
        target.kind === "llm" &&
        target.llm_type === "code"
      ) {
        nextEdges = addValidatorSemanticEdge(nextEdges, targetId, current.id);
      }
    }
  }

  return nextEdges;
}

export function syncSubcategoryConfigsForCategoryUpdate(
  current: NodeConfig,
  next: NodeConfig,
  configs: Record<string, NodeConfig>,
  oldName: string,
  newName: string,
  nameChanged: boolean,
): Record<string, NodeConfig> {
  if (!isCategoryConfig(current)) {
    return configs;
  }
  const nextCategory = isCategoryConfig(next) ? next : current;
  const oldValues = current.values ?? [];
  const newValues = nextCategory.values ?? [];
  const valuesChanged =
    oldValues.length !== newValues.length ||
    oldValues.some((value, index) => value !== newValues[index]);

  let nextConfigs = configs;
  for (const config of Object.values(configs)) {
    if (!isSubcategoryConfig(config)) {
      continue;
    }
    if (config.subcategory_parent !== oldName) {
      continue;
    }
    const mapping = config.subcategory_mapping ?? {};
    const nextMapping: Record<string, string[]> = {};
    for (const value of newValues) {
      nextMapping[value] = mapping[value] ?? [];
    }
    const updated: NodeConfig = {
      ...config,
      // biome-ignore lint/style/useNamingConvention: api schema
      subcategory_parent: nameChanged ? newName : config.subcategory_parent,
      // biome-ignore lint/style/useNamingConvention: api schema
      subcategory_mapping: valuesChanged ? nextMapping : mapping,
    };
    nextConfigs = { ...nextConfigs, [config.id]: updated };
  }
  return nextConfigs;
}
