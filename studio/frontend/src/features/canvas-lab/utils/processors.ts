import type { CanvasProcessorConfig } from "../types";

export function buildDefaultSchemaTransform(): CanvasProcessorConfig {
  return {
    id: "schema-transform-1",
    // biome-ignore lint/style/useNamingConvention: api schema
    processor_type: "schema_transform",
    name: "schema_transform",
    template: '{\n  "text": "{{ column_name }}"\n}',
  };
}
