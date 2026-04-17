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
    title: "生成字段",
    description: "通过列表、范围和可复用模式生成字段。",
    icon: DiceFaces03Icon,
  },
  {
    kind: "seed",
    title: "源数据",
    description: "从现有数据集或文件开始。",
    icon: Plant01Icon,
  },
  {
    kind: "llm",
    title: "AI 生成",
    description: "生成内容、连接模型并管理工具。",
    icon: PencilEdit02Icon,
  },
  {
    kind: "validator",
    title: "校验",
    description: "在配方流转中对生成代码进行检查或过滤。",
    icon: Shield02Icon,
  },
  {
    kind: "expression",
    title: "公式",
    description: "基于其它字段构建新字段。",
    icon: FunctionIcon,
  },
  {
    kind: "note",
    title: "笔记",
    description: "添加 Markdown 笔记以记录流程。",
    icon: PencilEdit02Icon,
  },
];

const BLOCK_DEFINITIONS: BlockDefinition[] = [
  {
    kind: "seed",
    type: "seed_hf",
    title: "Hugging Face 数据集",
    description: "使用 Hugging Face 数据集中的行作为源数据。",
    icon: Plant01Icon,
    dialogKey: "seed",
    createConfig: (id, existing) => makeSeedConfig(id, existing, "hf"),
  },
  {
    kind: "seed",
    type: "seed_local",
    title: "CSV 或 JSON 文件",
    description: "上传 CSV、JSON 或 JSONL，并将其行作为源数据。",
    icon: DocumentCodeIcon,
    dialogKey: "seed",
    createConfig: (id, existing) => makeSeedConfig(id, existing, "local"),
  },
  {
    kind: "seed",
    type: "seed_unstructured",
    title: "文档文件",
    description: "上传 PDF、DOCX 或 TXT，并转换为源数据行。",
    icon: DocumentAttachmentIcon,
    dialogKey: "seed",
    createConfig: (id, existing) => makeSeedConfig(id, existing, "unstructured"),
  },
  {
    kind: "sampler",
    type: "category",
    title: "类别",
    description: "从你定义的列表生成值，可选权重或规则。",
    icon: Tag01Icon,
    dialogKey: "category",
    createConfig: (id, existing) => makeSamplerConfig(id, "category", existing),
  },
  {
    kind: "sampler",
    type: "subcategory",
    title: "子类别",
    description: "基于你为每个类别定义的分组生成值。",
    icon: TagsIcon,
    dialogKey: "subcategory",
    createConfig: (id, existing) => makeSamplerConfig(id, "subcategory", existing),
  },
  {
    kind: "sampler",
    type: "uniform",
    title: "随机数",
    description: "在最小值与最大值之间生成数字。",
    icon: EqualSignIcon,
    dialogKey: "uniform",
    createConfig: (id, existing) => makeSamplerConfig(id, "uniform", existing),
  },
  {
    kind: "sampler",
    type: "gaussian",
    title: "正态分布数值",
    description: "围绕平均值生成数字。",
    icon: Parabola02Icon,
    dialogKey: "gaussian",
    createConfig: (id, existing) => makeSamplerConfig(id, "gaussian", existing),
  },
  {
    kind: "sampler",
    type: "bernoulli",
    title: "是/否值",
    description: "根据概率生成二元结果。",
    icon: EqualSignIcon,
    dialogKey: "bernoulli",
    createConfig: (id, existing) => makeSamplerConfig(id, "bernoulli", existing),
  },
  {
    kind: "sampler",
    type: "datetime",
    title: "日期时间",
    description: "在指定日期范围内生成时间戳。",
    icon: Clock01Icon,
    dialogKey: "datetime",
    createConfig: (id, existing) => makeSamplerConfig(id, "datetime", existing),
  },
  {
    kind: "sampler",
    type: "timedelta",
    title: "时间偏移",
    description: "基于其它日期字段生成时间差。",
    icon: Clock01Icon,
    dialogKey: "timedelta",
    createConfig: (id, existing) => makeSamplerConfig(id, "timedelta", existing),
  },
  {
    kind: "sampler",
    type: "uuid",
    title: "唯一 ID",
    description: "生成唯一标识符。",
    icon: FingerPrintIcon,
    dialogKey: "uuid",
    createConfig: (id, existing) => makeSamplerConfig(id, "uuid", existing),
  },
  {
    kind: "sampler",
    type: "person",
    title: "合成人物",
    description: "生成逼真的人物信息。",
    icon: UserAccountIcon,
    dialogKey: "person",
    createConfig: (id, existing) => makeSamplerConfig(id, "person", existing),
  },
  {
    kind: "llm",
    type: "text",
    title: "AI 文本",
    description: "根据提示词生成文本。",
    icon: PencilEdit02Icon,
    dialogKey: "llm",
    createConfig: (id, existing) => makeLlmConfig(id, "text", existing),
  },
  {
    kind: "llm",
    type: "structured",
    title: "AI 结构化数据",
    description: "生成符合响应格式的 JSON。",
    icon: CodeIcon,
    dialogKey: "llm",
    createConfig: (id, existing) => makeLlmConfig(id, "structured", existing),
  },
  {
    kind: "llm",
    type: "code",
    title: "AI 代码",
    description: "按你指定的语言生成代码。",
    icon: CodeSimpleIcon,
    dialogKey: "llm",
    createConfig: (id, existing) => makeLlmConfig(id, "code", existing),
  },
  {
    kind: "llm",
    type: "judge",
    title: "AI 评分器",
    description: "按你的标准对输出评分。",
    icon: BalanceScaleIcon,
    dialogKey: "llm",
    createConfig: (id, existing) => makeLlmConfig(id, "judge", existing),
  },
  {
    kind: "llm",
    type: "model_provider",
    title: "提供方连接",
    description: "选择模型请求去向与认证方式。",
    icon: Shield02Icon,
    dialogKey: "model_provider",
    createConfig: (id, existing) => makeModelProviderConfig(id, existing),
  },
  {
    kind: "llm",
    type: "model_config",
    title: "模型预设",
    description: "选择模型并保存可复用的生成设置。",
    icon: Plant01Icon,
    dialogKey: "model_config",
    createConfig: (id, existing) => makeModelConfig(id, existing),
  },
  {
    kind: "llm",
    type: "tool_config",
    title: "工具权限",
    description: "选择 AI 步骤可使用的工具。",
    icon: Plug01Icon,
    dialogKey: "tool_config",
    createConfig: (id, existing) => makeToolProfileConfig(id, existing),
  },
  {
    kind: "validator",
    type: "validator_python",
    title: "Python 校验",
    description: "检查生成的 Python，并过滤失败行。",
    icon: Shield02Icon,
    dialogKey: "validator",
    createConfig: (id, existing) =>
      makeValidatorConfig(id, "code", "python", existing),
  },
  {
    kind: "validator",
    type: "validator_sql",
    title: "SQL 校验",
    description: "检查生成的 SQL，并过滤失败行。",
    icon: Shield02Icon,
    dialogKey: "validator",
    createConfig: (id, existing) =>
      makeValidatorConfig(id, "code", "sql:sqlite", existing),
  },
  {
    kind: "validator",
    type: "validator_oxc",
    title: "JS/TS 校验",
    description: "检查生成的 JavaScript/TypeScript，并过滤失败行。",
    icon: Shield02Icon,
    dialogKey: "validator",
    createConfig: (id, existing) =>
      makeValidatorConfig(id, "oxc", "javascript", existing),
  },
  {
    kind: "expression",
    type: "expression",
    title: "公式",
    description: "使用其他字段构建或转换字段。",
    icon: FunctionIcon,
    dialogKey: "expression",
    createConfig: (id, existing) => makeExpressionConfig(id, existing),
  },
  {
    kind: "note",
    type: "markdown_note",
    title: "笔记",
    description: "在画布上添加说明。笔记不会影响运行结果。",
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
