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
  return {
    title: "LLM",
    kind: "llm",
    subtype: labelForLlm(config.llm_type),
    blockType: config.llm_type,
    name: config.name,
    layoutDirection,
  };
}
