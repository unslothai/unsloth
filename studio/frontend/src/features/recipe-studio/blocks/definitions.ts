// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  BalanceScaleIcon,
  Clock01Icon,
  CodeIcon,
  CodeSimpleIcon,
  DiceFaces03Icon,
  DocumentAttachmentIcon,
  DocumentCodeIcon,
  EqualSignIcon,
  FingerPrintIcon,
  FunctionIcon,
  Plug01Icon,
  Parabola02Icon,
  PencilEdit02Icon,
  Plant01Icon,
  Shield02Icon,
  Tag01Icon,
  TagsIcon,
  UserAccountIcon,
} from "@hugeicons/core-free-icons";
import type {
  LlmType,
  NodeConfig,
  SamplerType,
  SeedSourceType,
} from "../types";
import {
  makeExpressionConfig,
  makeLlmConfig,
  makeMarkdownNoteConfig,
  makeModelConfig,
  makeModelProviderConfig,
  makeToolProfileConfig,
  makeSamplerConfig,
  makeSeedConfig,
  makeValidatorConfig,
} from "../utils";

export type BlockKind =
  | "sampler"
  | "llm"
  | "validator"
  | "expression"
  | "seed"
  | "note";
export type BlockType =
  | SamplerType
  | LlmType
  | "validator_python"
  | "validator_sql"
  | "validator_oxc"
  | "expression"
  | "markdown_note"
  | "seed"
  | "seed_hf"
  | "seed_local"
  | "seed_unstructured"
  | "model_provider"
  | "model_config"
  | "tool_config";

export type SeedBlockType = "seed_hf" | "seed_local" | "seed_unstructured";

type IconType = typeof CodeIcon;

export type BlockGroup = {
  kind: BlockKind;
  title: string;
  description: string;
  icon: IconType;
};

export type BlockDialogKey =
  | "seed"
  | "markdown_note"
  | "category"
  | "subcategory"
  | "uniform"
  | "gaussian"
  | "bernoulli"
  | "datetime"
  | "timedelta"
  | "uuid"
  | "person"
  | "llm"
  | "validator"
  | "model_provider"
  | "model_config"
  | "tool_config"
  | "expression";

export type BlockDefinition = {
  kind: BlockKind;
  type: BlockType;
  title: string;
  description: string;
  icon: IconType;
  dialogKey: BlockDialogKey;
  createConfig: (id: string, existing: NodeConfig[]) => NodeConfig;
};

export const BLOCK_GROUPS: BlockGroup[] = [
  {
    kind: "sampler",
    title: "Samplers",
    description: "Fast deterministic columns from distributions and categories.",
    icon: DiceFaces03Icon,
  },
  {
    kind: "seed",
    title: "Seed",
    description: "Bootstrap generation from an existing dataset.",
    icon: Plant01Icon,
  },
  {
    kind: "llm",
    title: "LLM + Models",
    description: "Generation, model aliases, and shared tool profiles.",
    icon: PencilEdit02Icon,
  },
  {
    kind: "validator",
    title: "Validators",
    description: "Validate generated code outputs with built-in engines.",
    icon: Shield02Icon,
  },
  {
    kind: "expression",
    title: "Expression",
    description: "Derive columns with Jinja templates.",
    icon: FunctionIcon,
  },
  {
    kind: "note",
    title: "Notes",
    description: "Add markdown notes to document your flow.",
    icon: PencilEdit02Icon,
  },
];

const BLOCK_DEFINITIONS: BlockDefinition[] = [
  {
    kind: "seed",
    type: "seed_hf",
    title: "Hugging Face dataset",
    description: "Load real rows from HF and use them as generation context.",
    icon: Plant01Icon,
    dialogKey: "seed",
    createConfig: (id, existing) => makeSeedConfig(id, existing, "hf"),
  },
  {
    kind: "seed",
    type: "seed_local",
    title: "Structured file",
    description: "Upload CSV/JSON/JSONL and use rows as seed context.",
    icon: DocumentCodeIcon,
    dialogKey: "seed",
    createConfig: (id, existing) => makeSeedConfig(id, existing, "local"),
  },
  {
    kind: "seed",
    type: "seed_unstructured",
    title: "Unstructured documents",
    description: "Upload PDF/DOCX/TXT, chunk to text rows, then seed.",
    icon: DocumentAttachmentIcon,
    dialogKey: "seed",
    createConfig: (id, existing) => makeSeedConfig(id, existing, "unstructured"),
  },
  {
    kind: "sampler",
    type: "category",
    title: "Category",
    description: "Define categorical values with optional weights and conditions.",
    icon: Tag01Icon,
    dialogKey: "category",
    createConfig: (id, existing) => makeSamplerConfig(id, "category", existing),
  },
  {
    kind: "sampler",
    type: "subcategory",
    title: "Subcategory",
    description: "Define hierarchical values mapped to a parent category.",
    icon: TagsIcon,
    dialogKey: "subcategory",
    createConfig: (id, existing) => makeSamplerConfig(id, "subcategory", existing),
  },
  {
    kind: "sampler",
    type: "uniform",
    title: "Uniform",
    description: "Sample evenly between low and high.",
    icon: EqualSignIcon,
    dialogKey: "uniform",
    createConfig: (id, existing) => makeSamplerConfig(id, "uniform", existing),
  },
  {
    kind: "sampler",
    type: "gaussian",
    title: "Gaussian",
    description: "Sample from a normal distribution (mean/stddev).",
    icon: Parabola02Icon,
    dialogKey: "gaussian",
    createConfig: (id, existing) => makeSamplerConfig(id, "gaussian", existing),
  },
  {
    kind: "sampler",
    type: "bernoulli",
    title: "Bernoulli",
    description: "Sample binary outcomes from probability p.",
    icon: EqualSignIcon,
    dialogKey: "bernoulli",
    createConfig: (id, existing) => makeSamplerConfig(id, "bernoulli", existing),
  },
  {
    kind: "sampler",
    type: "datetime",
    title: "Datetime",
    description: "Sample timestamps within a start/end range.",
    icon: Clock01Icon,
    dialogKey: "datetime",
    createConfig: (id, existing) => makeSamplerConfig(id, "datetime", existing),
  },
  {
    kind: "sampler",
    type: "timedelta",
    title: "Timedelta",
    description: "Sample time offsets from a reference datetime column.",
    icon: Clock01Icon,
    dialogKey: "timedelta",
    createConfig: (id, existing) => makeSamplerConfig(id, "timedelta", existing),
  },
  {
    kind: "sampler",
    type: "uuid",
    title: "UUID",
    description: "Generate unique identifiers with optional formatting.",
    icon: FingerPrintIcon,
    dialogKey: "uuid",
    createConfig: (id, existing) => makeSamplerConfig(id, "uuid", existing),
  },
  {
    kind: "sampler",
    type: "person",
    title: "Person",
    description: "Generate realistic synthetic people with faker attributes.",
    icon: UserAccountIcon,
    dialogKey: "person",
    createConfig: (id, existing) => makeSamplerConfig(id, "person", existing),
  },
  {
    kind: "llm",
    type: "text",
    title: "LLM Text",
    description: "Generate natural language text from prompt templates.",
    icon: PencilEdit02Icon,
    dialogKey: "llm",
    createConfig: (id, existing) => makeLlmConfig(id, "text", existing),
  },
  {
    kind: "llm",
    type: "structured",
    title: "LLM Structured",
    description: "Generate JSON constrained to a schema.",
    icon: CodeIcon,
    dialogKey: "llm",
    createConfig: (id, existing) => makeLlmConfig(id, "structured", existing),
  },
  {
    kind: "llm",
    type: "code",
    title: "LLM Code",
    description: "Generate code in a chosen language with clean extraction.",
    icon: CodeSimpleIcon,
    dialogKey: "llm",
    createConfig: (id, existing) => makeLlmConfig(id, "code", existing),
  },
  {
    kind: "llm",
    type: "judge",
    title: "LLM Judge",
    description: "Score generated outputs with rubric-based criteria.",
    icon: BalanceScaleIcon,
    dialogKey: "llm",
    createConfig: (id, existing) => makeLlmConfig(id, "judge", existing),
  },
  {
    kind: "llm",
    type: "model_provider",
    title: "Model Provider",
    description: "Define endpoint and auth settings for model access.",
    icon: Shield02Icon,
    dialogKey: "model_provider",
    createConfig: (id, existing) => makeModelProviderConfig(id, existing),
  },
  {
    kind: "llm",
    type: "model_config",
    title: "Model Config",
    description: "Bind alias to model, provider, and inference settings.",
    icon: Plant01Icon,
    dialogKey: "model_config",
    createConfig: (id, existing) => makeModelConfig(id, existing),
  },
  {
    kind: "llm",
    type: "tool_config",
    title: "Tool Profile",
    description: "Reusable MCP servers + allowed tools for one or more LLMs.",
    icon: Plug01Icon,
    dialogKey: "tool_config",
    createConfig: (id, existing) => makeToolProfileConfig(id, existing),
  },
  {
    kind: "validator",
    type: "validator_python",
    title: "Python Validator",
    description: "Validate Python code columns.",
    icon: Shield02Icon,
    dialogKey: "validator",
    createConfig: (id, existing) =>
      makeValidatorConfig(id, "code", "python", existing),
  },
  {
    kind: "validator",
    type: "validator_sql",
    title: "SQL Validator",
    description: "Validate SQL code columns.",
    icon: Shield02Icon,
    dialogKey: "validator",
    createConfig: (id, existing) =>
      makeValidatorConfig(id, "code", "sql:sqlite", existing),
  },
  {
    kind: "validator",
    type: "validator_oxc",
    title: "OXC Validator",
    description: "Validate JavaScript or TypeScript code columns.",
    icon: Shield02Icon,
    dialogKey: "validator",
    createConfig: (id, existing) =>
      makeValidatorConfig(id, "oxc", "javascript", existing),
  },
  {
    kind: "expression",
    type: "expression",
    title: "Expression",
    description: "Transform/combine columns using Jinja expressions.",
    icon: FunctionIcon,
    dialogKey: "expression",
    createConfig: (id, existing) => makeExpressionConfig(id, existing),
  },
  {
    kind: "note",
    type: "markdown_note",
    title: "Markdown note",
    description: "UI-only markdown notes on canvas, not sent to backend.",
    icon: PencilEdit02Icon,
    dialogKey: "markdown_note",
    createConfig: (id, existing) => makeMarkdownNoteConfig(id, existing),
  },
];

export function getBlocksForKind(kind: BlockKind): BlockDefinition[] {
  return BLOCK_DEFINITIONS.filter((block) => block.kind === kind);
}

export function getBlockDefinition(
  kind: BlockKind,
  type: BlockType,
): BlockDefinition | null {
  return (
    BLOCK_DEFINITIONS.find((block) => block.kind === kind && block.type === type) ??
    null
  );
}

export function getBlockDefinitionForConfig(
  config: NodeConfig | null,
): BlockDefinition | null {
  if (!config) {
    return null;
  }
  if (config.kind === "seed") {
    const seedType: Record<SeedSourceType, SeedBlockType> = {
      hf: "seed_hf",
      local: "seed_local",
      unstructured: "seed_unstructured",
    };
    return getBlockDefinition("seed", seedType[config.seed_source_type ?? "hf"]);
  }
  if (config.kind === "sampler") {
    const samplerType =
      config.sampler_type === "person_from_faker" ? "person" : config.sampler_type;
    return getBlockDefinition("sampler", samplerType);
  }
  if (config.kind === "llm") {
    return getBlockDefinition("llm", config.llm_type);
  }
  if (config.kind === "validator") {
    if (config.validator_type === "oxc") {
      return getBlockDefinition("validator", "validator_oxc");
    }
    const isSql = config.code_lang.startsWith("sql:");
    return getBlockDefinition(
      "validator",
      isSql ? "validator_sql" : "validator_python",
    );
  }
  if (config.kind === "model_provider") {
    return getBlockDefinition("llm", "model_provider");
  }
  if (config.kind === "model_config") {
    return getBlockDefinition("llm", "model_config");
  }
  if (config.kind === "tool_config") {
    return getBlockDefinition("llm", "tool_config");
  }
  if (config.kind === "markdown_note") {
    return getBlockDefinition("note", "markdown_note");
  }
  return getBlockDefinition("expression", "expression");
}
