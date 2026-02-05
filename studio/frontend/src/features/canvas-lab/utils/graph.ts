import { type Connection, type Edge, addEdge } from "@xyflow/react";
import type { NodeConfig, SamplerConfig } from "../types";
import { HANDLE_IDS } from "./handles";
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

function isSemanticRelation(source: NodeConfig, target: NodeConfig): boolean {
  if (source.kind === "model_provider" && target.kind === "model_config") {
    return true;
  }
  return source.kind === "model_config" && target.kind === "llm";
}

function isModelInfraNode(config: NodeConfig): boolean {
  return config.kind === "model_provider" || config.kind === "model_config";
}

function isSemanticLane(connection: Connection): boolean {
  return (
    connection.sourceHandle === HANDLE_IDS.semanticOut &&
    connection.targetHandle === HANDLE_IDS.semanticIn
  );
}

function isDataLane(connection: Connection): boolean {
  return (
    connection.sourceHandle === HANDLE_IDS.dataOut &&
    connection.targetHandle === HANDLE_IDS.dataIn
  );
}

type SingleRefRelation =
  | "provider"
  | "model_alias"
  | "reference_column_name"
  | "subcategory_parent";

function getSingleRefRelation(
  source: NodeConfig,
  target: NodeConfig,
): SingleRefRelation | null {
  if (source.kind === "model_provider" && target.kind === "model_config") {
    return "provider";
  }
  if (source.kind === "model_config" && target.kind === "llm") {
    return "model_alias";
  }
  if (
    source.kind === "sampler" &&
    source.sampler_type === "datetime" &&
    target.kind === "sampler" &&
    target.sampler_type === "timedelta"
  ) {
    return "reference_column_name";
  }
  if (isCategoryConfig(source) && isSubcategoryConfig(target)) {
    return "subcategory_parent";
  }
  return null;
}

function isCompetingIncomingEdge(
  edge: Edge,
  targetId: string,
  relation: SingleRefRelation,
  configs: Record<string, NodeConfig>,
): boolean {
  if (edge.target !== targetId) {
    return false;
  }
  const source = configs[edge.source];
  if (!source) {
    return false;
  }
  if (relation === "provider") {
    return source.kind === "model_provider";
  }
  if (relation === "model_alias") {
    return source.kind === "model_config";
  }
  if (relation === "subcategory_parent") {
    return isCategoryConfig(source);
  }
  return source.kind === "sampler" && source.sampler_type === "datetime";
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
  if (!(source && target)) {
    return false;
  }
  const semanticRelation = isSemanticRelation(source, target);
  if (semanticRelation) {
    return isSemanticLane(connection);
  }
  if (isModelInfraNode(source) || isModelInfraNode(target)) {
    return false;
  }
  return isDataLane(connection);
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
  const semanticRelation = isSemanticRelation(source, target);
  const singleRefRelation = getSingleRefRelation(source, target);
  const nextBaseEdges =
    singleRefRelation
      ? edges.filter(
          (edge) =>
            !isCompetingIncomingEdge(
              edge,
              target.id,
              singleRefRelation,
              configs,
            ),
        )
      : edges;
  const nextEdges = addEdge(
    { ...connection, type: semanticRelation ? "semantic" : "canvas" },
    nextBaseEdges,
  );
  if (source.kind === "model_provider" && target.kind === "model_config") {
    const next = { ...target, provider: source.name };
    return { edges: nextEdges, configs: { ...configs, [target.id]: next } };
  }
  if (source.kind === "model_config" && target.kind === "llm") {
    const next = { ...target, model_alias: source.name };
    return { edges: nextEdges, configs: { ...configs, [target.id]: next } };
  }
  if (
    source.kind === "sampler" &&
    source.sampler_type === "datetime" &&
    target.kind === "sampler" &&
    target.sampler_type === "timedelta"
  ) {
    const next = {
      ...target,
      // biome-ignore lint/style/useNamingConvention: api schema
      reference_column_name: source.name,
    };
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
