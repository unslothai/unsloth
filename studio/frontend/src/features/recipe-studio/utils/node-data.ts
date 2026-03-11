// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

import type { RecipeNodeData, LayoutDirection, NodeConfig } from "../types";
import {
  labelForExpression,
  labelForLlm,
  labelForSampler,
} from "./config-labels";

export function nodeDataFromConfig(
  config: NodeConfig,
  layoutDirection: LayoutDirection = "LR",
): RecipeNodeData {
  if (config.kind === "sampler") {
    return {
      title: "Sampler",
      kind: "sampler",
      subtype: labelForSampler(config.sampler_type),
      blockType: config.sampler_type,
      name: config.name,
      layoutDirection,
    };
  }
  if (config.kind === "expression") {
    return {
      title: "Expression",
      kind: "expression",
      subtype: labelForExpression(config.dtype),
      blockType: "expression",
      name: config.name,
      layoutDirection,
    };
  }
  if (config.kind === "validator") {
    const isOxc = config.validator_type === "oxc";
    const isSql = config.code_lang.startsWith("sql:");
    let subtype = "Python";
    let blockType: RecipeNodeData["blockType"] = "validator_python";
    if (isOxc) {
      subtype = "OXC";
      blockType = "validator_oxc";
    } else if (isSql) {
      subtype = "SQL";
      blockType = "validator_sql";
    }
    return {
      title: "Validator",
      kind: "validator",
      subtype,
      blockType,
      name: config.name,
      layoutDirection,
    };
  }
  if (config.kind === "markdown_note") {
    return {
      title: "Note",
      kind: "note",
      subtype: "Markdown",
      blockType: "markdown_note",
      name: config.name,
      layoutDirection,
    };
  }
  if (config.kind === "seed") {
    const seedSourceType = config.seed_source_type ?? "hf";
    const sourceLabel =
      seedSourceType === "hf"
        ? "Hugging Face dataset"
        : seedSourceType === "local"
          ? "Structured file"
          : "Unstructured document";
    return {
      title: "Seed",
      kind: "seed",
      subtype: sourceLabel,
      blockType: "seed",
      name: sourceLabel,
      layoutDirection,
    };
  }
  if (config.kind === "model_provider") {
    return {
      title: "Model Provider",
      kind: "model_provider",
      subtype: config.provider_type || "Provider",
      blockType: "model_provider",
      name: config.name,
      layoutDirection,
    };
  }
  if (config.kind === "model_config") {
    return {
      title: "Model Config",
      kind: "model_config",
      subtype: config.model || "Model",
      blockType: "model_config",
      name: config.name,
      layoutDirection,
    };
  }
  if (config.kind === "tool_config") {
    const providerCount = config.mcp_providers.length;
    return {
      title: "Tool Profile",
      kind: "tool_config",
      subtype: providerCount === 1 ? "1 MCP server" : `${providerCount} MCP servers`,
      blockType: "tool_config",
      name: config.name,
      layoutDirection,
    };
  }
  return {
    title: "LLM",
    kind: "llm",
    subtype: labelForLlm(config.llm_type),
    blockType: config.llm_type,
    name: config.name,
    layoutDirection,
  };
}
