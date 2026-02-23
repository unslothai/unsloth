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
  Parabola02Icon,
  PencilEdit02Icon,
  Plant01Icon,
  Shield02Icon,
  Tag01Icon,
  TagsIcon,
  UserAccountIcon,
} from "@hugeicons/core-free-icons";
import type { LlmType, NodeConfig, SamplerType, SeedSourceType } from "../types";
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
  | "seed_hf"
  | "seed_local"
  | "seed_unstructured"
  | "model_provider"
  | "model_config";

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
    description: "Generation, providers, and model aliases.",
    icon: PencilEdit02Icon,
  },
  {
    kind: "expression",
    title: "Expression",
    description: "Derive columns with Jinja templates.",
    icon: FunctionIcon,
  },
];

const BLOCK_DEFINITIONS: BlockDefinition[] = [
  {
    kind: "seed",
    type: "seed_hf",
    title: "Hugginface dataset",
    description: "Load real rows from HF and use them as generation context.",
    icon: Plant01Icon,
    dialogKey: "seed",
    createConfig: (id, existing) => makeSeedConfig(id, existing, "hf"),
  },
  {
    kind: "seed",
    type: "seed_local",
    title: "Local file",
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
    kind: "expression",
    type: "expression",
    title: "Expression",
    description: "Transform/combine columns using Jinja expressions.",
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
  if (config.kind === "model_provider") {
    return getBlockDefinition("llm", "model_provider");
  }
  if (config.kind === "model_config") {
    return getBlockDefinition("llm", "model_config");
  }
  return getBlockDefinition("expression", "expression");
}
