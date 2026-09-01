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
  GithubIcon,
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
  | "seed_github"
  | "model_provider"
  | "model_config"
  | "tool_config";

export type SeedBlockType =
  | "seed_hf"
  | "seed_local"
  | "seed_unstructured"
  | "seed_github";

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
    title: "Generated fields",
    description: "Create fields from lists, ranges, and reusable patterns.",
    icon: DiceFaces03Icon,
  },
  {
    kind: "seed",
    title: "Source data",
    description: "Start from an existing dataset or file.",
    icon: Plant01Icon,
  },
  {
    kind: "llm",
    title: "AI generation",
    description: "Generate content, connect models, and manage tools.",
    icon: PencilEdit02Icon,
  },
  {
    kind: "validator",
    title: "Checks",
    description: "Lint or filter generated code as it moves through the recipe.",
    icon: Shield02Icon,
  },
  {
    kind: "expression",
    title: "Formulas",
    description: "Build a field from other fields.",
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
    description: "Use rows from a Hugging Face dataset as source data.",
    icon: Plant01Icon,
    dialogKey: "seed",
    createConfig: (id, existing) => makeSeedConfig(id, existing, "hf"),
  },
  {
    kind: "seed",
    type: "seed_local",
    title: "CSV or JSON file",
    description: "Upload CSV, JSON, or JSONL and use its rows as source data.",
    icon: DocumentCodeIcon,
    dialogKey: "seed",
    createConfig: (id, existing) => makeSeedConfig(id, existing, "local"),
  },
  {
    kind: "seed",
    type: "seed_unstructured",
    title: "Document file",
    description: "Upload PDF, DOCX, or TXT and turn it into source rows.",
    icon: DocumentAttachmentIcon,
    dialogKey: "seed",
    createConfig: (id, existing) => makeSeedConfig(id, existing, "unstructured"),
  },
  {
    kind: "seed",
    type: "seed_github",
    title: "GitHub repositories",
    description: "Crawl issues, pull requests, and commits from one or more GitHub repos.",
    icon: GithubIcon,
    dialogKey: "seed",
    createConfig: (id, existing) => makeSeedConfig(id, existing, "github_repo"),
  },
  {
    kind: "sampler",
    type: "category",
    title: "Category",
    description: "Generate values from a list you define, with optional weights or rules.",
    icon: Tag01Icon,
    dialogKey: "category",
    createConfig: (id, existing) => makeSamplerConfig(id, "category", existing),
  },
  {
    kind: "sampler",
    type: "subcategory",
    title: "Subcategory",
    description: "Generate values from groups you define for each category.",
    icon: TagsIcon,
    dialogKey: "subcategory",
    createConfig: (id, existing) => makeSamplerConfig(id, "subcategory", existing),
  },
  {
    kind: "sampler",
    type: "uniform",
    title: "Random number",
    description: "Generate a number anywhere between a minimum and maximum.",
    icon: EqualSignIcon,
    dialogKey: "uniform",
    createConfig: (id, existing) => makeSamplerConfig(id, "uniform", existing),
  },
  {
    kind: "sampler",
    type: "gaussian",
    title: "Bell-curve number",
    description: "Generate numbers around an average value.",
    icon: Parabola02Icon,
    dialogKey: "gaussian",
    createConfig: (id, existing) => makeSamplerConfig(id, "gaussian", existing),
  },
  {
    kind: "sampler",
    type: "bernoulli",
    title: "Yes/no value",
    description: "Generate a binary result from a probability.",
    icon: EqualSignIcon,
    dialogKey: "bernoulli",
    createConfig: (id, existing) => makeSamplerConfig(id, "bernoulli", existing),
  },
  {
    kind: "sampler",
    type: "datetime",
    title: "Date and time",
    description: "Generate timestamps inside a date range.",
    icon: Clock01Icon,
    dialogKey: "datetime",
    createConfig: (id, existing) => makeSamplerConfig(id, "datetime", existing),
  },
  {
    kind: "sampler",
    type: "timedelta",
    title: "Time offset",
    description: "Generate a time difference from another date field.",
    icon: Clock01Icon,
    dialogKey: "timedelta",
    createConfig: (id, existing) => makeSamplerConfig(id, "timedelta", existing),
  },
  {
    kind: "sampler",
    type: "uuid",
    title: "Unique ID",
    description: "Generate unique identifiers.",
    icon: FingerPrintIcon,
    dialogKey: "uuid",
    createConfig: (id, existing) => makeSamplerConfig(id, "uuid", existing),
  },
  {
    kind: "sampler",
    type: "person",
    title: "Synthetic person",
    description: "Generate realistic person details.",
    icon: UserAccountIcon,
    dialogKey: "person",
    createConfig: (id, existing) => makeSamplerConfig(id, "person", existing),
  },
  {
    kind: "llm",
    type: "text",
    title: "AI text",
    description: "Generate text from your prompt.",
    icon: PencilEdit02Icon,
    dialogKey: "llm",
    createConfig: (id, existing) => makeLlmConfig(id, "text", existing),
  },
  {
    kind: "llm",
    type: "structured",
    title: "AI structured data",
    description: "Generate JSON that follows a response format.",
    icon: CodeIcon,
    dialogKey: "llm",
    createConfig: (id, existing) => makeLlmConfig(id, "structured", existing),
  },
  {
    kind: "llm",
    type: "code",
    title: "AI code",
    description: "Generate code in the language you choose.",
    icon: CodeSimpleIcon,
    dialogKey: "llm",
    createConfig: (id, existing) => makeLlmConfig(id, "code", existing),
  },
  {
    kind: "llm",
    type: "judge",
    title: "AI scorer",
    description: "Score outputs against your criteria.",
    icon: BalanceScaleIcon,
    dialogKey: "llm",
    createConfig: (id, existing) => makeLlmConfig(id, "judge", existing),
  },
  {
    kind: "llm",
    type: "model_provider",
    title: "Provider connection",
    description: "Choose where model requests go and how to sign in.",
    icon: Shield02Icon,
    dialogKey: "model_provider",
    createConfig: (id, existing) => makeModelProviderConfig(id, existing),
  },
  {
    kind: "llm",
    type: "model_config",
    title: "Model preset",
    description: "Pick a model and save reusable generation settings.",
    icon: Plant01Icon,
    dialogKey: "model_config",
    createConfig: (id, existing) => makeModelConfig(id, existing),
  },
  {
    kind: "llm",
    type: "tool_config",
    title: "Tool access",
    description: "Choose which tools an AI step can use.",
    icon: Plug01Icon,
    dialogKey: "tool_config",
    createConfig: (id, existing) => makeToolProfileConfig(id, existing),
  },
  {
    kind: "validator",
    type: "validator_python",
    title: "Python check",
    description: "Lint generated Python and filter out rows that fail.",
    icon: Shield02Icon,
    dialogKey: "validator",
    createConfig: (id, existing) =>
      makeValidatorConfig(id, "code", "python", existing),
  },
  {
    kind: "validator",
    type: "validator_sql",
    title: "SQL check",
    description: "Lint generated SQL and filter out rows that fail.",
    icon: Shield02Icon,
    dialogKey: "validator",
    createConfig: (id, existing) =>
      makeValidatorConfig(id, "code", "sql:sqlite", existing),
  },
  {
    kind: "validator",
    type: "validator_oxc",
    title: "JS/TS check",
    description: "Lint generated JavaScript or TypeScript and filter out rows that fail.",
    icon: Shield02Icon,
    dialogKey: "validator",
    createConfig: (id, existing) =>
      makeValidatorConfig(id, "oxc", "javascript", existing),
  },
  {
    kind: "expression",
    type: "expression",
    title: "Formula",
    description: "Build or transform a field using other fields.",
    icon: FunctionIcon,
    dialogKey: "expression",
    createConfig: (id, existing) => makeExpressionConfig(id, existing),
  },
  {
    kind: "note",
    type: "markdown_note",
    title: "Note",
    description: "Add a note to the canvas. Notes do not affect the run.",
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
      github_repo: "seed_github",
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
