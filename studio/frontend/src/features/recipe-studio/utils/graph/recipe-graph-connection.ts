// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type Connection, type Edge, addEdge } from "@xyflow/react";
import type { LayoutDirection, NodeConfig, SamplerConfig } from "../../types";
import {
  HANDLE_IDS,
  isDataSourceHandle,
  isDataTargetHandle,
  isSemanticSourceHandle,
  isSemanticTargetHandle,
  normalizeRecipeHandleId,
} from "../handles";
import { isSemanticRelation } from "./relations";
import {
  isCategoryConfig,
  isExpressionConfig,
  isSubcategoryConfig,
} from "../index";
import {
  VALIDATOR_OXC_CODE_LANGS,
  VALIDATOR_SQL_CODE_LANGS,
} from "../validators/code-lang";

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

function isModelInfraNode(config: NodeConfig): boolean {
  return (
    config.kind === "model_provider" ||
    config.kind === "model_config" ||
    config.kind === "tool_config"
  );
}

function isSemanticLane(connection: Connection): boolean {
  return (
    (isSemanticSourceHandle(connection.sourceHandle) ||
      isDataSourceHandle(connection.sourceHandle)) &&
    (isSemanticTargetHandle(connection.targetHandle) ||
      isDataTargetHandle(connection.targetHandle))
  );
}

function isDataLane(connection: Connection): boolean {
  return (
    isDataSourceHandle(connection.sourceHandle) &&
    isDataTargetHandle(connection.targetHandle)
  );
}

type SingleRefRelation =
  | "provider"
  | "model_alias"
  | "tool_alias"
  | "reference_column_name"
  | "subcategory_parent"
  | "validator_target_columns";

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
  if (source.kind === "tool_config" && target.kind === "llm") {
    return "tool_alias";
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
  if (
    source.kind === "llm" &&
    source.llm_type === "code" &&
    target.kind === "validator"
  ) {
    return "validator_target_columns";
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
  if (relation === "tool_alias") {
    return source.kind === "tool_config";
  }
  if (relation === "subcategory_parent") {
    return isCategoryConfig(source);
  }
  if (relation === "validator_target_columns") {
    return source.kind === "llm" && source.llm_type === "code";
  }
  return source.kind === "sampler" && source.sampler_type === "datetime";
}

function isModelSemanticRelation(source: NodeConfig, target: NodeConfig): boolean {
  return (
    (source.kind === "model_provider" && target.kind === "model_config") ||
    (source.kind === "model_config" && target.kind === "llm") ||
    (source.kind === "tool_config" && target.kind === "llm")
  );
}

function canApplyCodeLangToValidator(
  validator: Extract<NodeConfig, { kind: "validator" }>,
  codeLang: string,
): boolean {
  const normalized = codeLang.trim();
  if (!normalized) {
    return false;
  }
  if (validator.validator_type === "oxc") {
    return VALIDATOR_OXC_CODE_LANGS.includes(
      normalized as typeof validator.code_lang,
    );
  }
  if (normalized === "python") {
    return true;
  }
  return VALIDATOR_SQL_CODE_LANGS.includes(normalized as typeof validator.code_lang);
}

function countHandleUsage(
  edges: Edge[],
  nodeId: string,
  handleId: string,
  lane: "source" | "target",
): number {
  return edges.reduce((count, edge) => {
    const edgeNodeId = lane === "source" ? edge.source : edge.target;
    if (edgeNodeId !== nodeId) {
      return count;
    }
    const edgeHandleId =
      lane === "source"
        ? normalizeRecipeHandleId(edge.sourceHandle)
        : normalizeRecipeHandleId(edge.targetHandle);
    return edgeHandleId === handleId ? count + 1 : count;
  }, 0);
}

function pickLeastUsedHandle(
  candidates: string[],
  requested: string | null,
  usageFor: (handleId: string) => number,
): string {
  let bestHandle = candidates[0];
  let bestCount = Number.POSITIVE_INFINITY;
  const requestedNormalized = requested
    ? normalizeRecipeHandleId(requested)
    : null;

  for (const candidate of candidates) {
    const usage = usageFor(candidate);
    if (usage < bestCount) {
      bestHandle = candidate;
      bestCount = usage;
      continue;
    }
    if (usage === bestCount && requestedNormalized === candidate) {
      bestHandle = candidate;
    }
  }

  return bestHandle;
}

function chooseModelSemanticHandles(
  connection: Connection,
  source: NodeConfig,
  target: NodeConfig,
  edges: Edge[],
  layoutDirection: LayoutDirection,
): Connection {
  if (!isModelSemanticRelation(source, target)) {
    return connection;
  }

  const sourceCandidates =
    source.kind === "model_config" && target.kind === "llm"
      ? layoutDirection === "TB"
        ? [HANDLE_IDS.semanticOut]
        : [HANDLE_IDS.semanticOutBottom]
      : layoutDirection === "TB"
        ? [HANDLE_IDS.semanticOut, HANDLE_IDS.semanticOutBottom]
        : [HANDLE_IDS.semanticOutBottom, HANDLE_IDS.semanticOut];
  const targetCandidates =
    target.kind === "model_config"
      ? layoutDirection === "TB"
        ? [HANDLE_IDS.semanticIn, HANDLE_IDS.semanticInTop]
        : [HANDLE_IDS.semanticInTop, HANDLE_IDS.semanticIn]
      : [
          HANDLE_IDS.dataInTop,
          HANDLE_IDS.dataInBottom,
          HANDLE_IDS.dataIn,
          HANDLE_IDS.dataInRight,
        ];

  const sourceHandle = pickLeastUsedHandle(
    sourceCandidates,
    connection.sourceHandle ?? null,
    (handleId) => countHandleUsage(edges, source.id, handleId, "source"),
  );
  const targetHandle = pickLeastUsedHandle(
    targetCandidates,
    connection.targetHandle ?? null,
    (handleId) => countHandleUsage(edges, target.id, handleId, "target"),
  );

  return {
    ...connection,
    sourceHandle,
    targetHandle,
  };
}

function normalizeValidatorSemanticConnection(
  connection: Connection,
  source: NodeConfig,
  target: NodeConfig,
): Connection {
  if (
    source.kind === "validator" &&
    target.kind === "llm" &&
    target.llm_type === "code"
  ) {
    return {
      ...connection,
      source: target.id,
      target: source.id,
      sourceHandle: HANDLE_IDS.dataOut,
      targetHandle: HANDLE_IDS.dataIn,
    };
  }
  return connection;
}

export function isValidRecipeConnection(
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

export function applyRecipeConnection(
  connection: Connection,
  configs: Record<string, NodeConfig>,
  edges: Edge[],
  layoutDirection: LayoutDirection = "LR",
): { edges: Edge[]; configs?: Record<string, NodeConfig> } {
  if (!isValidRecipeConnection(connection, configs)) {
    return { edges };
  }
  const initialSource = connection.source
    ? configs[connection.source]
    : null;
  const initialTarget = connection.target
    ? configs[connection.target]
    : null;
  if (!(initialSource && initialTarget)) {
    return { edges };
  }
  const normalizedConnection = normalizeValidatorSemanticConnection(
    connection,
    initialSource,
    initialTarget,
  );
  const source = normalizedConnection.source
    ? configs[normalizedConnection.source]
    : null;
  const target = normalizedConnection.target
    ? configs[normalizedConnection.target]
    : null;
  if (!(source && target)) {
    return { edges };
  }

  const semanticRelation = isSemanticRelation(source, target);
  const singleRefRelation = getSingleRefRelation(source, target);
  if (
    singleRefRelation === "subcategory_parent" &&
    isSubcategoryConfig(target)
  ) {
    const currentParent = target.subcategory_parent?.trim() ?? "";
    if (currentParent && currentParent !== source.name) {
      return { edges };
    }
  }
  const nextBaseEdges = singleRefRelation
    ? edges.filter(
        (edge) =>
          !isCompetingIncomingEdge(edge, target.id, singleRefRelation, configs),
      )
    : edges;
  const resolvedConnection = chooseModelSemanticHandles(
    normalizedConnection,
    source,
    target,
    nextBaseEdges,
    layoutDirection,
  );
  const nextEdges = addEdge(
    { ...resolvedConnection, type: semanticRelation ? "semantic" : "canvas" },
    nextBaseEdges,
  );
  if (source.kind === "model_provider" && target.kind === "model_config") {
    // Keep the model_config.model field in sync with provider mode when the
    // link is changed via graph drag (the model-config dialog path has its
    // own applyProviderChange helper that does the same thing).
    const isSourceLocal = source.is_local === true;
    let nextModel = target.model;
    if (isSourceLocal && !nextModel.trim()) {
      nextModel = "local";
    } else if (!isSourceLocal && nextModel === "local") {
      nextModel = "";
    }
    const next = { ...target, provider: source.name, model: nextModel };
    return { edges: nextEdges, configs: { ...configs, [target.id]: next } };
  }
  if (source.kind === "model_config" && target.kind === "llm") {
    const next = { ...target, model_alias: source.name };
    return { edges: nextEdges, configs: { ...configs, [target.id]: next } };
  }
  if (source.kind === "tool_config" && target.kind === "llm") {
    const next = { ...target, tool_alias: source.name };
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
  if (
    source.kind === "llm" &&
    source.llm_type === "code" &&
    target.kind === "validator"
  ) {
    const nextCodeLang = (source.code_lang ?? "").trim();
    const canUseCodeLangForTarget = canApplyCodeLangToValidator(
      target,
      nextCodeLang,
    );
    const next = {
      ...target,
      // biome-ignore lint/style/useNamingConvention: api schema
      target_columns: [source.name],
      // biome-ignore lint/style/useNamingConvention: api schema
      code_lang:
        (
          canUseCodeLangForTarget ? nextCodeLang : target.code_lang
        ) as typeof target.code_lang,
    };
    return { edges: nextEdges, configs: { ...configs, [target.id]: next } };
  }
  if (
    isExpressionConfig(target) &&
    !semanticRelation &&
    source.kind !== "seed" &&
    source.kind !== "model_provider" &&
    source.kind !== "model_config" &&
    source.kind !== "validator"
  ) {
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
