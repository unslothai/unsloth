import type { NodeConfig } from "../../types";

export function isSemanticRelation(
  source: NodeConfig,
  target: NodeConfig,
): boolean {
  if (source.kind === "model_provider" && target.kind === "model_config") {
    return true;
  }
  return source.kind === "model_config" && target.kind === "llm";
}

