import { type Connection, type Edge, addEdge } from "@xyflow/react";
import type { NodeConfig, SamplerConfig } from "../types";
import {
  isCategoryConfig,
  isExpressionConfig,
  isLlmConfig,
  isSubcategoryConfig,
} from "./index";

function buildTemplateWithRef(template: string, ref: string): string {
  if (template.includes(ref)) {
    return template;
  }
  if (template.trim()) {
    return `${template}\n${ref}`;
  }
  return ref;
}

function syncSubcategoryMapping(
  subcategory: SamplerConfig,
  parent: NodeConfig,
): SamplerConfig {
  if (!isCategoryConfig(parent)) {
    return {
      ...subcategory,
      // biome-ignore lint/style/useNamingConvention: api schema
      subcategory_parent: parent.name,
    };
  }
  const nextMapping: Record<string, string[]> = {
    ...(subcategory.subcategory_mapping ?? {}),
  };
  for (const value of parent.values ?? []) {
    if (!nextMapping[value]) {
      nextMapping[value] = [];
    }
  }
  return {
    ...subcategory,
    // biome-ignore lint/style/useNamingConvention: api schema
    subcategory_parent: parent.name,
    // biome-ignore lint/style/useNamingConvention: api schema
    subcategory_mapping: nextMapping,
  };
}

function isSemanticEdge(source: NodeConfig, target: NodeConfig): boolean {
  if (source.kind === "model_provider" && target.kind === "model_config") {
    return true;
  }
  if (source.kind === "model_config" && target.kind === "llm") {
    return true;
  }
  return (
    source.kind === "sampler" &&
    source.sampler_type === "category" &&
    target.kind === "sampler" &&
    target.sampler_type === "subcategory"
  );
}

export function isValidCanvasConnection(
  connection: Connection,
  configs: Record<string, NodeConfig>,
): boolean {
  if (!(connection.source && connection.target)) {
    return false;
  }
  if (connection.source === connection.target) {
    return false;
  }
  const source = configs[connection.source];
  const target = configs[connection.target];
  return Boolean(source && target);
}

export function applyCanvasConnection(
  connection: Connection,
  configs: Record<string, NodeConfig>,
  edges: Edge[],
): { edges: Edge[]; configs?: Record<string, NodeConfig> } {
  if (!isValidCanvasConnection(connection, configs)) {
    return { edges };
  }
  const source = connection.source ? configs[connection.source] : null;
  const target = connection.target ? configs[connection.target] : null;
  if (!(source && target)) {
    return { edges };
  }
  const nextEdges = addEdge(
    { ...connection, type: isSemanticEdge(source, target) ? "semantic" : "canvas" },
    edges,
  );
  if (source.kind === "model_provider" && target.kind === "model_config") {
    const next = { ...target, provider: source.name };
    return { edges: nextEdges, configs: { ...configs, [target.id]: next } };
  }
  if (source.kind === "model_config" && target.kind === "llm") {
    const next = { ...target, model_alias: source.name };
    return { edges: nextEdges, configs: { ...configs, [target.id]: next } };
  }
  if (isLlmConfig(target) && source.kind !== "model_provider" && source.kind !== "model_config") {
    const ref = `{{ ${source.name} }}`;
    const next = {
      ...target,
      prompt: buildTemplateWithRef(target.prompt ?? "", ref),
    };
    return { edges: nextEdges, configs: { ...configs, [target.id]: next } };
  }
  if (isExpressionConfig(target) && source.kind !== "model_provider" && source.kind !== "model_config") {
    const ref = `{{ ${source.name} }}`;
    const next = {
      ...target,
      expr: buildTemplateWithRef(target.expr ?? "", ref),
    };
    return { edges: nextEdges, configs: { ...configs, [target.id]: next } };
  }
  if (isSubcategoryConfig(target) && isCategoryConfig(source)) {
    const next = syncSubcategoryMapping(target, source);
    return { edges: nextEdges, configs: { ...configs, [target.id]: next } };
  }
  return { edges: nextEdges };
}
