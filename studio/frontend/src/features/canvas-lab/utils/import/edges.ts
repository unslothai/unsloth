import type { Edge } from "@xyflow/react";
import type { NodeConfig } from "../../types";
import { extractRefs } from "./helpers";

function isSemanticConnection(source: NodeConfig, target: NodeConfig): boolean {
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

export function buildEdges(
  configs: NodeConfig[],
  nameToId: Map<string, string>,
  uiEdges: Array<{ from: string; to: string; type?: string }> | null,
): Edge[] {
  const edges: Edge[] = [];
  const seen = new Set<string>();
  const configByName = new Map(configs.map((config) => [config.name, config]));
  const addEdgeByName = (from: string, to: string, type?: string) => {
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
    const normalizedType =
      source && target && isSemanticConnection(source, target)
        ? "semantic"
        : (type ?? "canvas");
    edges.push({
      id: `e-${key}`,
      source: sourceId,
      target: targetId,
      type: normalizedType,
    });
  };

  if (uiEdges && uiEdges.length > 0) {
    for (const edge of uiEdges) {
      addEdgeByName(edge.from, edge.to, edge.type);
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
      addEdgeByName(config.subcategory_parent, config.name, "semantic");
    }
    if (config.kind === "model_config" && config.provider) {
      addEdgeByName(config.provider, config.name, "semantic");
    }
    if (config.kind === "llm" && config.model_alias) {
      addEdgeByName(config.model_alias, config.name, "semantic");
    }
  }

  return edges;
}
