import {
  BalanceScaleIcon,
  Clock01Icon,
  CodeIcon,
  CodeSimpleIcon,
  DiceFaces03Icon,
  EqualSignIcon,
  FingerPrintIcon,
  FunctionIcon,
  Parabola02Icon,
  PencilEdit02Icon,
  Plant01Icon,
  Shield02Icon,
  Tag01Icon,
  TagsIcon,
  UserAccountIcon,
} from "@hugeicons/core-free-icons";
import type { LlmType, NodeConfig, SamplerType } from "../types";
import {
  makeExpressionConfig,
  makeLlmConfig,
  makeModelConfig,
  makeModelProviderConfig,
  makeSamplerConfig,
  makeSeedConfig,
} from "../utils";

export type BlockKind = "sampler" | "llm" | "expression" | "seed";
export type BlockType =
  | SamplerType
  | LlmType
  | "expression"
  | "seed"
  | "model_provider"
  | "model_config";

type IconType = typeof CodeIcon;

export type BlockGroup = {
  kind: BlockKind;
  title: string;
  description: string;
  icon: IconType;
};

export type BlockDialogKey =
  | "seed"
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
  | "model_provider"
  | "model_config"
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
    description: "Numeric + categorical blocks.",
    icon: DiceFaces03Icon,
  },
  {
    kind: "seed",
    title: "Seed",
    description: "Columns from a seed dataset.",
    icon: Plant01Icon,
  },
  {
    kind: "llm",
    title: "LLM + Models",
    description: "Generation, providers, and model aliases.",
    icon: PencilEdit02Icon,
  },
  {
    kind: "expression",
    title: "Expression",
    description: "Derived columns with Jinja.",
    icon: FunctionIcon,
  },
];

const BLOCK_DEFINITIONS: BlockDefinition[] = [
  {
    kind: "seed",
    type: "seed",
    title: "Seed (Hugging Face)",
    description: "Configure a HF seed dataset.",
    icon: Plant01Icon,
    dialogKey: "seed",
    createConfig: (id, existing) => makeSeedConfig(id, existing),
  },
  {
    kind: "sampler",
    type: "category",
    title: "Category",
    description: "Pick from a list of values.",
    icon: Tag01Icon,
    dialogKey: "category",
    createConfig: (id, existing) => makeSamplerConfig(id, "category", existing),
  },
  {
    kind: "sampler",
    type: "subcategory",
    title: "Subcategory",
    description: "Map sub-values to a category.",
    icon: TagsIcon,
    dialogKey: "subcategory",
    createConfig: (id, existing) => makeSamplerConfig(id, "subcategory", existing),
  },
  {
    kind: "sampler",
    type: "uniform",
    title: "Uniform",
    description: "Random number between low/high.",
    icon: EqualSignIcon,
    dialogKey: "uniform",
    createConfig: (id, existing) => makeSamplerConfig(id, "uniform", existing),
  },
  {
    kind: "sampler",
    type: "gaussian",
    title: "Gaussian",
    description: "Normal distribution sampler.",
    icon: Parabola02Icon,
    dialogKey: "gaussian",
    createConfig: (id, existing) => makeSamplerConfig(id, "gaussian", existing),
  },
  {
    kind: "sampler",
    type: "bernoulli",
    title: "Bernoulli",
    description: "Binary sampler with probability.",
    icon: EqualSignIcon,
    dialogKey: "bernoulli",
    createConfig: (id, existing) => makeSamplerConfig(id, "bernoulli", existing),
  },
  {
    kind: "sampler",
    type: "datetime",
    title: "Datetime",
    description: "Date/time range sampler.",
    icon: Clock01Icon,
    dialogKey: "datetime",
    createConfig: (id, existing) => makeSamplerConfig(id, "datetime", existing),
  },
  {
    kind: "sampler",
    type: "timedelta",
    title: "Timedelta",
    description: "Offset from datetime column.",
    icon: Clock01Icon,
    dialogKey: "timedelta",
    createConfig: (id, existing) => makeSamplerConfig(id, "timedelta", existing),
  },
  {
    kind: "sampler",
    type: "uuid",
    title: "UUID",
    description: "UUID string sampler.",
    icon: FingerPrintIcon,
    dialogKey: "uuid",
    createConfig: (id, existing) => makeSamplerConfig(id, "uuid", existing),
  },
  {
    kind: "sampler",
    type: "person",
    title: "Person",
    description: "Faker person sampler.",
    icon: UserAccountIcon,
    dialogKey: "person",
    createConfig: (id, existing) => makeSamplerConfig(id, "person", existing),
  },
  {
    kind: "llm",
    type: "text",
    title: "LLM Text",
    description: "Free-form prompt generation.",
    icon: PencilEdit02Icon,
    dialogKey: "llm",
    createConfig: (id, existing) => makeLlmConfig(id, "text", existing),
  },
  {
    kind: "llm",
    type: "structured",
    title: "LLM Structured",
    description: "JSON output via schema.",
    icon: CodeIcon,
    dialogKey: "llm",
    createConfig: (id, existing) => makeLlmConfig(id, "structured", existing),
  },
  {
    kind: "llm",
    type: "code",
    title: "LLM Code",
    description: "Generate code or SQL.",
    icon: CodeSimpleIcon,
    dialogKey: "llm",
    createConfig: (id, existing) => makeLlmConfig(id, "code", existing),
  },
  {
    kind: "llm",
    type: "judge",
    title: "LLM Judge",
    description: "Score outputs with criteria.",
    icon: BalanceScaleIcon,
    dialogKey: "llm",
    createConfig: (id, existing) => makeLlmConfig(id, "judge", existing),
  },
  {
    kind: "llm",
    type: "model_provider",
    title: "Model Provider",
    description: "Endpoint, auth, and provider settings.",
    icon: Shield02Icon,
    dialogKey: "model_provider",
    createConfig: (id, existing) => makeModelProviderConfig(id, existing),
  },
  {
    kind: "llm",
    type: "model_config",
    title: "Model Config",
    description: "Alias, model, provider, and inference params.",
    icon: Plant01Icon,
    dialogKey: "model_config",
    createConfig: (id, existing) => makeModelConfig(id, existing),
  },
  {
    kind: "expression",
    type: "expression",
    title: "Expression",
    description: "Transform columns with Jinja.",
    icon: FunctionIcon,
    dialogKey: "expression",
    createConfig: (id, existing) => makeExpressionConfig(id, existing),
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
    return getBlockDefinition("seed", "seed");
  }
  if (config.kind === "sampler") {
    const samplerType =
      config.sampler_type === "person_from_faker" ? "person" : config.sampler_type;
    return getBlockDefinition("sampler", samplerType);
  }
  if (config.kind === "llm") {
    return getBlockDefinition("llm", config.llm_type);
  }
  if (config.kind === "model_provider") {
    return getBlockDefinition("llm", "model_provider");
  }
  if (config.kind === "model_config") {
    return getBlockDefinition("llm", "model_config");
  }
  return getBlockDefinition("expression", "expression");
}
