// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { Edge } from "@xyflow/react";
import type { LayoutDirection, NodeConfig } from "../../types";
import {
  getDefaultDataSourceHandle,
  getDefaultDataTargetHandle,
  getDefaultSemanticSourceHandle,
  getDefaultSemanticTargetHandle,
  isDataSourceHandle,
  isDataTargetHandle,
  isSemanticSourceHandle,
  isSemanticTargetHandle,
  normalizeRecipeHandleId,
} from "../handles";
import { extractRefs } from "./helpers";

function isSemanticConnection(source: NodeConfig, target: NodeConfig): boolean {
  if (source.kind === "model_provider" && target.kind === "model_config") {
    return true;
  }
  if (source.kind === "model_config" && target.kind === "llm") {
    return true;
  }
  if (source.kind === "tool_config" && target.kind === "llm") {
    return true;
  }
  if (
    source.kind === "llm" &&
    source.llm_type === "code" &&
    target.kind === "validator"
  ) {
    return true;
  }
  return (
    source.kind === "validator" &&
    target.kind === "llm" &&
    target.llm_type === "code"
  );
}

export function buildEdges(
  configs: NodeConfig[],
  nameToId: Map<string, string>,
  uiEdges:
    | Array<{
        from: string;
        to: string;
        type?: string;
        sourceHandle?: string;
        targetHandle?: string;
      }>
    | null,
  layoutDirection: LayoutDirection,
): Edge[] {
  const edges: Edge[] = [];
  const seen = new Set<string>();
  const configByName = new Map(configs.map((config) => [config.name, config]));
  const addEdgeByName = (
    from: string,
    to: string,
    sourceHandleInput?: string,
    targetHandleInput?: string,
  ): void => {
    const sourceId = nameToId.get(from);
    const targetId = nameToId.get(to);
    if (!(sourceId && targetId)) {
      return;
    }
    const key = `${sourceId}-${targetId}`;
    if (seen.has(key)) {
      return;
    }
    seen.add(key);
    const source = configByName.get(from);
    const target = configByName.get(to);
    const isSemantic = Boolean(
      source && target && isSemanticConnection(source, target),
    );
    const normalizedType = isSemantic ? "semantic" : "canvas";
    const sourceHandleNormalized = normalizeRecipeHandleId(sourceHandleInput);
    const targetHandleNormalized = normalizeRecipeHandleId(targetHandleInput);
    const semanticSourceDefault =
      source?.kind === "llm"
        ? getDefaultDataSourceHandle(layoutDirection)
        : getDefaultSemanticSourceHandle(layoutDirection);
    const semanticTargetDefault =
      target?.kind === "llm"
        ? getDefaultDataTargetHandle(layoutDirection)
        : getDefaultSemanticTargetHandle(layoutDirection);
    let sourceHandle = getDefaultDataSourceHandle(layoutDirection);
    let targetHandle = getDefaultDataTargetHandle(layoutDirection);

    if (isSemantic) {
      sourceHandle =
        isSemanticSourceHandle(sourceHandleNormalized) ||
        isDataSourceHandle(sourceHandleNormalized)
          ? sourceHandleNormalized ?? semanticSourceDefault
          : semanticSourceDefault;
      targetHandle =
        isSemanticTargetHandle(targetHandleNormalized) ||
        isDataTargetHandle(targetHandleNormalized)
          ? targetHandleNormalized ?? semanticTargetDefault
          : semanticTargetDefault;
    } else {
      sourceHandle = isDataSourceHandle(sourceHandleNormalized)
        ? sourceHandleNormalized ?? getDefaultDataSourceHandle(layoutDirection)
        : getDefaultDataSourceHandle(layoutDirection);
      targetHandle = isDataTargetHandle(targetHandleNormalized)
        ? targetHandleNormalized ?? getDefaultDataTargetHandle(layoutDirection)
        : getDefaultDataTargetHandle(layoutDirection);
    }
    edges.push({
      id: `e-${key}`,
      source: sourceId,
      target: targetId,
      type: normalizedType,
      sourceHandle,
      targetHandle,
    });
  };

  if (uiEdges && uiEdges.length > 0) {
    for (const edge of uiEdges) {
      addEdgeByName(
        edge.from,
        edge.to,
        edge.sourceHandle,
        edge.targetHandle,
      );
    }
    if (edges.length > 0) {
      return edges;
    }
  }

  for (const config of configs) {
    if (config.kind === "llm") {
      for (const ref of extractRefs(config.prompt ?? "")) {
        addEdgeByName(ref, config.name);
      }
      for (const ref of extractRefs(config.system_prompt ?? "")) {
        addEdgeByName(ref, config.name);
      }
    }
    if (config.kind === "expression") {
      for (const ref of extractRefs(config.expr)) {
        addEdgeByName(ref, config.name);
      }
    }
    if (
      config.kind === "sampler" &&
      config.sampler_type === "subcategory" &&
      config.subcategory_parent
    ) {
      addEdgeByName(config.subcategory_parent, config.name);
    }
    if (config.kind === "model_config" && config.provider) {
      addEdgeByName(config.provider, config.name);
    }
    if (
      config.kind === "sampler" &&
      config.sampler_type === "timedelta" &&
      config.reference_column_name
    ) {
      addEdgeByName(config.reference_column_name, config.name);
    }
    if (config.kind === "llm" && config.model_alias) {
      addEdgeByName(config.model_alias, config.name);
    }
    if (config.kind === "llm" && config.tool_alias) {
      addEdgeByName(config.tool_alias, config.name);
    }
    if (config.kind === "validator") {
      for (const targetColumn of config.target_columns ?? []) {
        if (targetColumn.trim()) {
          addEdgeByName(targetColumn, config.name);
        }
      }
    }
  }

  return edges;
}
