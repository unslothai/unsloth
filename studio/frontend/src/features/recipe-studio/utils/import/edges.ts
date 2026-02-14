import type { Edge } from "@xyflow/react";
import type { NodeConfig } from "../../types";
import { HANDLE_IDS } from "../handles";
import { extractRefs } from "./helpers";

function isSemanticConnection(source: NodeConfig, target: NodeConfig): boolean {
  if (source.kind === "model_provider" && target.kind === "model_config") {
    return true;
  }
  return source.kind === "model_config" && target.kind === "llm";
}

export function buildEdges(
  configs: NodeConfig[],
  nameToId: Map<string, string>,
  uiEdges: Array<{ from: string; to: string; type?: string }> | null,
): Edge[] {
  const edges: Edge[] = [];
  const seen = new Set<string>();
  const configByName = new Map(configs.map((config) => [config.name, config]));
  const addEdgeByName = (from: string, to: string) => {
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
    const isSemantic = Boolean(source && target && isSemanticConnection(source, target));
    const normalizedType = isSemantic ? "semantic" : "canvas";
    const handles =
      normalizedType === "semantic"
        ? {
            sourceHandle: HANDLE_IDS.semanticOut,
            targetHandle: HANDLE_IDS.semanticIn,
          }
        : {
            sourceHandle: HANDLE_IDS.dataOut,
            targetHandle: HANDLE_IDS.dataIn,
          };
    edges.push({
      id: `e-${key}`,
      source: sourceId,
      target: targetId,
      type: normalizedType,
      ...handles,
    });
  };

  if (uiEdges && uiEdges.length > 0) {
    for (const edge of uiEdges) {
      addEdgeByName(edge.from, edge.to);
    }
    return edges;
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
  }

  return edges;
}
