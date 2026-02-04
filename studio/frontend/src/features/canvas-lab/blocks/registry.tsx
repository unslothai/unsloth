import {
  CodeIcon,
  Database02Icon,
  Flowchart01Icon,
  SparklesIcon,
} from "@hugeicons/core-free-icons";
import type { ReactElement } from "react";
import type { LlmType, NodeConfig, SamplerConfig, SamplerType } from "../types";
import {
  makeExpressionConfig,
  makeLlmConfig,
  makeSamplerConfig,
} from "../utils";
import { ExpressionDialog } from "../dialogs/expression/expression-dialog";
import { LlmDialog } from "../dialogs/llm/llm-dialog";
import { CategoryDialog } from "../dialogs/samplers/category-dialog";
import { DatetimeDialog } from "../dialogs/samplers/datetime-dialog";
import { GaussianDialog } from "../dialogs/samplers/gaussian-dialog";
import { PersonDialog } from "../dialogs/samplers/person-dialog";
import { SubcategoryDialog } from "../dialogs/samplers/subcategory-dialog";
import { UniformDialog } from "../dialogs/samplers/uniform-dialog";
import { UuidDialog } from "../dialogs/samplers/uuid-dialog";

export type BlockKind = "sampler" | "llm" | "expression";
export type BlockType = SamplerType | LlmType | "expression";

type IconType = typeof Database02Icon;

type BlockGroup = {
  kind: BlockKind;
  title: string;
  description: string;
  icon: IconType;
};

type BlockDialogArgs = {
  config: NodeConfig;
  categoryOptions: SamplerConfig[];
  onUpdate: (id: string, patch: Partial<NodeConfig>) => void;
};

type BlockDefinition = {
  kind: BlockKind;
  type: BlockType;
  title: string;
  description: string;
  icon: IconType;
  createConfig: (id: string, existing: NodeConfig[]) => NodeConfig;
  renderDialog: (args: BlockDialogArgs) => ReactElement | null;
};

export const BLOCK_GROUPS: BlockGroup[] = [
  {
    kind: "sampler",
    title: "Sampler",
    description: "Numeric + categorical blocks.",
    icon: Database02Icon,
  },
  {
    kind: "llm",
    title: "LLM",
    description: "Text + structured blocks.",
    icon: SparklesIcon,
  },
  {
    kind: "expression",
    title: "Expression",
    description: "Derived columns with Jinja.",
    icon: CodeIcon,
  },
];

const BLOCK_DEFINITIONS: BlockDefinition[] = [
  {
    kind: "sampler",
    type: "category",
    title: "Category",
    description: "Pick from a list of values.",
    icon: Database02Icon,
    createConfig: (id, existing) => makeSamplerConfig(id, "category", existing),
    renderDialog: ({ config, onUpdate }) =>
      config.kind === "sampler" && config.sampler_type === "category" ? (
        <CategoryDialog
          config={config}
          onUpdate={(patch) => onUpdate(config.id, patch)}
        />
      ) : null,
  },
  {
    kind: "sampler",
    type: "subcategory",
    title: "Subcategory",
    description: "Map sub-values to a category.",
    icon: Database02Icon,
    createConfig: (id, existing) =>
      makeSamplerConfig(id, "subcategory", existing),
    renderDialog: ({ config, categoryOptions, onUpdate }) =>
      config.kind === "sampler" && config.sampler_type === "subcategory" ? (
        <SubcategoryDialog
          config={config}
          categoryOptions={categoryOptions}
          onUpdate={(patch) => onUpdate(config.id, patch)}
        />
      ) : null,
  },
  {
    kind: "sampler",
    type: "uniform",
    title: "Uniform",
    description: "Random number between low/high.",
    icon: Database02Icon,
    createConfig: (id, existing) => makeSamplerConfig(id, "uniform", existing),
    renderDialog: ({ config, onUpdate }) =>
      config.kind === "sampler" && config.sampler_type === "uniform" ? (
        <UniformDialog
          config={config}
          onUpdate={(patch) => onUpdate(config.id, patch)}
        />
      ) : null,
  },
  {
    kind: "sampler",
    type: "gaussian",
    title: "Gaussian",
    description: "Normal distribution sampler.",
    icon: Database02Icon,
    createConfig: (id, existing) => makeSamplerConfig(id, "gaussian", existing),
    renderDialog: ({ config, onUpdate }) =>
      config.kind === "sampler" && config.sampler_type === "gaussian" ? (
        <GaussianDialog
          config={config}
          onUpdate={(patch) => onUpdate(config.id, patch)}
        />
      ) : null,
  },
  {
    kind: "sampler",
    type: "datetime",
    title: "Datetime",
    description: "Date/time range sampler.",
    icon: Database02Icon,
    createConfig: (id, existing) => makeSamplerConfig(id, "datetime", existing),
    renderDialog: ({ config, onUpdate }) =>
      config.kind === "sampler" && config.sampler_type === "datetime" ? (
        <DatetimeDialog
          config={config}
          onUpdate={(patch) => onUpdate(config.id, patch)}
        />
      ) : null,
  },
  {
    kind: "sampler",
    type: "uuid",
    title: "UUID",
    description: "UUID string sampler.",
    icon: Database02Icon,
    createConfig: (id, existing) => makeSamplerConfig(id, "uuid", existing),
    renderDialog: ({ config, onUpdate }) =>
      config.kind === "sampler" && config.sampler_type === "uuid" ? (
        <UuidDialog
          config={config}
          onUpdate={(patch) => onUpdate(config.id, patch)}
        />
      ) : null,
  },
  {
    kind: "sampler",
    type: "person",
    title: "Person",
    description: "Synthetic person sampler.",
    icon: Database02Icon,
    createConfig: (id, existing) => makeSamplerConfig(id, "person", existing),
    renderDialog: ({ config, onUpdate }) =>
      config.kind === "sampler" &&
      (config.sampler_type === "person" ||
        config.sampler_type === "person_from_faker") ? (
        <PersonDialog
          config={config}
          onUpdate={(patch) => onUpdate(config.id, patch)}
        />
      ) : null,
  },
  {
    kind: "llm",
    type: "text",
    title: "LLM Text",
    description: "Free-form prompt generation.",
    icon: SparklesIcon,
    createConfig: (id, existing) => makeLlmConfig(id, "text", existing),
    renderDialog: ({ config, onUpdate }) =>
      config.kind === "llm" && config.llm_type === "text" ? (
        <LlmDialog
          config={config}
          onUpdate={(patch) => onUpdate(config.id, patch)}
        />
      ) : null,
  },
  {
    kind: "llm",
    type: "structured",
    title: "LLM Structured",
    description: "JSON output via schema.",
    icon: Flowchart01Icon,
    createConfig: (id, existing) => makeLlmConfig(id, "structured", existing),
    renderDialog: ({ config, onUpdate }) =>
      config.kind === "llm" && config.llm_type === "structured" ? (
        <LlmDialog
          config={config}
          onUpdate={(patch) => onUpdate(config.id, patch)}
        />
      ) : null,
  },
  {
    kind: "llm",
    type: "code",
    title: "LLM Code",
    description: "Generate code or SQL.",
    icon: CodeIcon,
    createConfig: (id, existing) => makeLlmConfig(id, "code", existing),
    renderDialog: ({ config, onUpdate }) =>
      config.kind === "llm" && config.llm_type === "code" ? (
        <LlmDialog
          config={config}
          onUpdate={(patch) => onUpdate(config.id, patch)}
        />
      ) : null,
  },
  {
    kind: "expression",
    type: "expression",
    title: "Expression",
    description: "Transform columns with Jinja.",
    icon: CodeIcon,
    createConfig: (id, existing) => makeExpressionConfig(id, existing),
    renderDialog: ({ config, onUpdate }) =>
      config.kind === "expression" ? (
        <ExpressionDialog
          config={config}
          onUpdate={(patch) => onUpdate(config.id, patch)}
        />
      ) : null,
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
    BLOCK_DEFINITIONS.find(
      (block) => block.kind === kind && block.type === type,
    ) ?? null
  );
}

export function getBlockDefinitionForConfig(
  config: NodeConfig | null,
): BlockDefinition | null {
  if (!config) {
    return null;
  }
  if (config.kind === "sampler") {
    const samplerType =
      config.sampler_type === "person_from_faker"
        ? "person"
        : config.sampler_type;
    return getBlockDefinition("sampler", samplerType);
  }
  if (config.kind === "llm") {
    return getBlockDefinition("llm", config.llm_type);
  }
  return getBlockDefinition("expression", "expression");
}

export function renderBlockDialog(
  config: NodeConfig | null,
  categoryOptions: SamplerConfig[],
  onUpdate: (id: string, patch: Partial<NodeConfig>) => void,
): ReactElement | null {
  const definition = getBlockDefinitionForConfig(config);
  if (!definition || !config) {
    return null;
  }
  return definition.renderDialog({ config, categoryOptions, onUpdate });
}
