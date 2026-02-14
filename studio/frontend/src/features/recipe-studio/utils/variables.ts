import type { NodeConfig } from "../types";

export type AvailableRefItem = {
  ref: string;
  kind: Exclude<NodeConfig["kind"], "model_provider" | "model_config">;
  subtype: string;
  valueType?: string;
};

function getStructuredRefs(
  llmName: string,
  outputFormat: string,
): Array<{ ref: string; valueType?: string }> {
  try {
    const schema = JSON.parse(outputFormat);
    if (!(schema?.properties && typeof schema.properties === "object")) {
      return [];
    }
    return Object.keys(schema.properties).map((key) => {
      const prop = schema.properties[key];
      const valueType =
        prop && typeof prop === "object" && typeof prop.type === "string"
          ? prop.type
          : undefined;
      return { ref: `${llmName}.${key}`, valueType };
    });
  } catch {
    return [];
  }
}

export function getAvailableRefItems(
  configs: Record<string, NodeConfig>,
  currentId: string,
): AvailableRefItem[] {
  const items: AvailableRefItem[] = [];

  for (const config of Object.values(configs)) {
    if (config.id === currentId) continue;
    if (config.kind === "model_provider" || config.kind === "model_config") continue;

    if (config.kind === "sampler") {
      items.push({ ref: config.name, kind: "sampler", subtype: config.sampler_type });
      continue;
    }

    if (config.kind === "expression") {
      items.push({ ref: config.name, kind: "expression", subtype: config.dtype });
      continue;
    }

    if (config.kind === "llm") {
      items.push({ ref: config.name, kind: "llm", subtype: config.llm_type });
      if (config.llm_type === "structured" && config.output_format) {
        for (const ref of getStructuredRefs(config.name, config.output_format)) {
          items.push({
            ref: ref.ref,
            kind: "llm",
            subtype: config.llm_type,
            valueType: ref.valueType,
          });
        }
      }
    }
  }

  return items;
}

export function getAvailableVariables(
  configs: Record<string, NodeConfig>,
  currentId: string,
): string[] {
  return getAvailableRefItems(configs, currentId).map((item) => item.ref);
}

