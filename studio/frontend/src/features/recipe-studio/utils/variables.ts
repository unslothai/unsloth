import type { NodeConfig } from "../types";

export function getAvailableVariables(
  configs: Record<string, NodeConfig>,
  currentId: string,
): string[] {
  const vars: string[] = [];
  for (const config of Object.values(configs)) {
    if (config.id === currentId) continue;
    if (config.kind === "model_provider" || config.kind === "model_config") continue;
    vars.push(config.name);
    if (config.kind === "llm" && config.llm_type === "structured" && config.output_format) {
      try {
        const schema = JSON.parse(config.output_format);
        if (schema.properties) {
          for (const key of Object.keys(schema.properties)) {
            vars.push(`${config.name}.${key}`);
          }
        }
      } catch {
        /* skip invalid JSON */
      }
    }
  }
  return vars;
}
