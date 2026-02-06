import type { CanvasProcessorConfig, NodeConfig } from "../../types";
import { buildEdges } from "./edges";
import { isRecord, parseJson, readString } from "./helpers";
import {
  parseColumn,
  parseModelConfig,
  parseModelProvider,
} from "./parsers";
import { buildNodes, parseUi } from "./ui";
import type { ImportResult } from "./types";

type RecipeInput = {
  columns?: unknown;
  model_configs?: unknown;
  model_providers?: unknown;
  processors?: unknown;
};

type UiInput = {
  nodes?: unknown;
  edges?: unknown;
};

function parseProcessors(input: unknown): CanvasProcessorConfig[] {
  if (!Array.isArray(input)) {
    return [];
  }
  const processors: CanvasProcessorConfig[] = [];
  input.forEach((item, index) => {
    if (!isRecord(item)) {
      return;
    }
    const type = readString(item.processor_type);
    const templateRaw = item.template;
    const isSchemaTransform =
      type === "schema_transform" || isRecord(templateRaw);
    if (!isSchemaTransform) {
      return;
    }
    const name = readString(item.name) ?? `schema_transform_${index + 1}`;
    const template =
      typeof templateRaw === "string"
        ? templateRaw
        : isRecord(templateRaw)
          ? JSON.stringify(templateRaw, null, 2)
          : "{\n  \"text\": \"{{ column_name }}\"\n}";
    processors.push({
      id: `p${index + 1}`,
      // biome-ignore lint/style/useNamingConvention: api schema
      processor_type: "schema_transform",
      name,
      template,
    });
  });
  return processors;
}

export function importCanvasPayload(input: string): ImportResult {
  const parsed = parseJson(input);
  if (!parsed.data || !isRecord(parsed.data)) {
    return {
      errors: [parsed.error ?? "Invalid JSON payload."],
      snapshot: null,
    };
  }

  const recipe = (isRecord(parsed.data.recipe)
    ? parsed.data.recipe
    : parsed.data) as RecipeInput;
  const ui = isRecord(parsed.data.ui) ? (parsed.data.ui as UiInput) : null;

  if (!Array.isArray(recipe.columns)) {
    return { errors: ["Recipe must include columns."], snapshot: null };
  }

  const errors: string[] = [];
  const configs: NodeConfig[] = [];
  const processors = parseProcessors(recipe.processors);
  const nameToId = new Map<string, string>();

  let nextId = 1;

  if (Array.isArray(recipe.model_providers)) {
    recipe.model_providers.forEach((provider, index) => {
      if (!isRecord(provider)) {
        errors.push(`Model provider ${index + 1}: invalid object.`);
        return;
      }
      const name = readString(provider.name);
      if (!name) {
        errors.push(`Model provider ${index + 1}: missing name.`);
        return;
      }
      const id = `n${nextId}`;
      nextId += 1;
      const config = parseModelProvider(provider, name, id);
      if (nameToId.has(config.name)) {
        errors.push(`Duplicate column name: ${config.name}.`);
        return;
      }
      nameToId.set(config.name, config.id);
      configs.push(config);
    });
  }

  if (Array.isArray(recipe.model_configs)) {
    recipe.model_configs.forEach((model, index) => {
      if (!isRecord(model)) {
        errors.push(`Model config ${index + 1}: invalid object.`);
        return;
      }
      const name = readString(model.alias) ?? readString(model.name);
      if (!name) {
        errors.push(`Model config ${index + 1}: missing alias.`);
        return;
      }
      const id = `n${nextId}`;
      nextId += 1;
      const config = parseModelConfig(model, name, id);
      if (nameToId.has(config.name)) {
        errors.push(`Duplicate column name: ${config.name}.`);
        return;
      }
      nameToId.set(config.name, config.id);
      configs.push(config);
    });
  }

  recipe.columns.forEach((column, index) => {
    if (!isRecord(column)) {
      errors.push(`Column ${index + 1}: invalid object.`);
      return;
    }
    const id = `n${nextId}`;
    nextId += 1;
    const config = parseColumn(column, id, errors);
    if (!config) {
      return;
    }
    if (nameToId.has(config.name)) {
      errors.push(`Duplicate column name: ${config.name}.`);
      return;
    }
    nameToId.set(config.name, config.id);
    configs.push(config);
  });

  if (errors.length > 0) {
    return { errors, snapshot: null };
  }

  const { layouts, edges: uiEdges } = parseUi(ui);
  const nodes = buildNodes(configs, layouts);
  const edges = buildEdges(configs, nameToId, uiEdges);

  const maxY = nodes.reduce(
    (acc, node) => Math.max(acc, node.position.y),
    0,
  );

  return {
    errors: [],
    snapshot: {
      configs: Object.fromEntries(configs.map((config) => [config.id, config])),
      nodes,
      edges,
      processors,
      nextId,
      nextY: maxY + 140,
    },
  };
}
