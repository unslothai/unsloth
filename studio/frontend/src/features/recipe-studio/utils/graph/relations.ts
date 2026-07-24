// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { NodeConfig } from "../../types";

export function isSemanticRelation(
  source: NodeConfig,
  target: NodeConfig,
): boolean {
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

/** Relation labels keyed by `${source.kind}>${target.kind}`. */
const SEMANTIC_EDGE_LABELS: Record<string, string> = {
  "model_provider>model_config": "provider",
  "model_config>llm": "model",
  "tool_config>llm": "tools",
  "llm>validator": "code",
  "validator>llm": "scores",
};

/**
 * Short net label for a wire, schematic-style (like D0 / Out). Semantic links
 * read as their relation ("provider", "model", "tools", "scores", "code");
 * data links read as the upstream signal they carry (the source node's name).
 */
export function edgeSignalLabel(
  source: NodeConfig | undefined,
  target: NodeConfig | undefined,
): string {
  if (source && target) {
    const relation = SEMANTIC_EDGE_LABELS[`${source.kind}>${target.kind}`];
    if (relation) {
      return relation;
    }
  }
  return source?.name ?? "";
}
