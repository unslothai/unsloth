import type { NodeConfig } from "../types";

function getStructuredRefs(llmName: string, outputFormat: string): string[] {
  try {
    const schema = JSON.parse(outputFormat);
    if (!(schema?.properties && typeof schema.properties === "object")) {
      return [];
    }
    return Object.keys(schema.properties).map((key) => `${llmName}.${key}`);
  } catch {
    return [];
  }
}

export function getAvailableVariables(
  configs: Record<string, NodeConfig>,
  currentId: string,
): string[] {
  const vars: string[] = [];

  for (const config of Object.values(configs)) {
    if (config.id === currentId) {
      continue;
    }
    if (config.kind === "model_provider" || config.kind === "model_config") {
      continue;
    }

    if (config.kind === "sampler") {
      vars.push(config.name);
      continue;
    }

    if (config.kind === "expression") {
      vars.push(config.name);
      continue;
    }

    if (config.kind === "seed") {
      for (const col of config.seed_columns ?? []) {
        const name = col.trim();
        if (!name) continue;
        vars.push(name);
      }
      continue;
    }

    if (config.kind !== "llm") {
      continue;
    }

    vars.push(config.name);
    if (config.llm_type !== "structured" || !config.output_format) {
      continue;
    }
    vars.push(...getStructuredRefs(config.name, config.output_format));
  }

  return vars;
}
