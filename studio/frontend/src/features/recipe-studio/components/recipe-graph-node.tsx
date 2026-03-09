// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { MarkdownPreview } from "@/components/markdown/markdown-preview";
import { cn } from "@/lib/utils";
import {
  BalanceScaleIcon,
  Clock01Icon,
  CodeIcon,
  CodeSimpleIcon,
  DiceFaces03Icon,
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
import { HugeiconsIcon } from "@hugeicons/react";
import {
  NodeResizer,
  Position,
  useUpdateNodeInternals,
  type NodeProps,
} from "@xyflow/react";
import { memo, type ReactElement, useEffect } from "react";
import { MAX_NODE_WIDTH, MAX_NOTE_NODE_WIDTH, MIN_NODE_WIDTH } from "../constants";
import { useRecipeStudioStore } from "../stores/recipe-studio";
import type {
  RecipeNode as RecipeGraphNodeType,
  LlmType,
  NodeConfig,
  SamplerType,
} from "../types";
import { NODE_HANDLE_CLASS } from "../utils/handle-layout";
import { HANDLE_IDS } from "../utils/handles";
import { InlineCategoryBadges } from "./inline/inline-category-badges";
import { InlineExpression } from "./inline/inline-expression";
import { InlineLlm } from "./inline/inline-llm";
import { InlineModel } from "./inline/inline-model";
import { isInlineConfig } from "./inline/inline-policy";
import { InlineSampler } from "./inline/inline-sampler";
import { InlineSeed } from "./inline/inline-seed";
import {
  BaseNode,
  BaseNodeContent,
  BaseNodeHeader,
  BaseNodeHeaderTitle,
} from "./rf-ui/base-node";
import { LabeledHandle } from "./rf-ui/labeled-handle";

type IconType = typeof CodeIcon;

function hexToRgb(hex: string): { r: number; g: number; b: number } | null {
  const normalized = hex.trim().replace("#", "");
  if (!/^[0-9a-fA-F]{6}$/.test(normalized)) {
    return null;
  }
  const int = Number.parseInt(normalized, 16);
  return {
    r: (int >> 16) & 255,
    g: (int >> 8) & 255,
    b: int & 255,
  };
}

function parseNoteOpacity(value: string | undefined): number {
  const parsed = Number.parseInt(value ?? "", 10);
  if (!Number.isFinite(parsed)) {
    return 0.35;
  }
  return Math.max(0.05, Math.min(1, parsed / 100));
}

const NODE_META = {
  sampler: {
    tone: "bg-emerald-50 text-emerald-600 border-emerald-100",
  },
  llm: {
    tone: "bg-sky-50 text-sky-600 border-sky-100",
  },
  validator: {
    tone: "bg-rose-50 text-rose-600 border-rose-100",
  },
  expression: {
    tone: "bg-indigo-50 text-indigo-600 border-indigo-100",
  },
  note: {
    tone: "bg-violet-50 text-violet-700 border-violet-100",
  },
  seed: {
    tone: "bg-lime-50 text-lime-700 border-lime-100",
  },
  model_provider: {
    tone: "bg-amber-50 text-amber-600 border-amber-100",
  },
  model_config: {
    tone: "bg-orange-50 text-orange-600 border-orange-100",
  },
  tool_config: {
    tone: "bg-cyan-50 text-cyan-700 border-cyan-100",
  },
} as const;
const USER_NODE_TONE =
  "bg-amber-50 text-amber-700 border-amber-100 dark:bg-amber-950/30 dark:text-amber-300 dark:border-amber-900/60";

const SAMPLER_ICONS: Record<SamplerType, IconType> = {
  category: Tag01Icon,
  subcategory: TagsIcon,
  uniform: EqualSignIcon,
  gaussian: Parabola02Icon,
  bernoulli: EqualSignIcon,
  datetime: Clock01Icon,
  timedelta: Clock01Icon,
  uuid: FingerPrintIcon,
  person: UserAccountIcon,
  person_from_faker: UserAccountIcon,
};

const LLM_ICONS: Record<LlmType, IconType> = {
  text: PencilEdit02Icon,
  structured: CodeIcon,
  code: CodeSimpleIcon,
  judge: BalanceScaleIcon,
};

function resolveNodeIcon(
  kind: RecipeGraphNodeType["data"]["kind"],
  blockType: RecipeGraphNodeType["data"]["blockType"],
): IconType {
  if (kind === "sampler" && blockType in SAMPLER_ICONS) {
    return SAMPLER_ICONS[blockType as SamplerType];
  }
  if (kind === "llm" && blockType in LLM_ICONS) {
    return LLM_ICONS[blockType as LlmType];
  }
  if (kind === "validator") {
    return Shield02Icon;
  }
  if (kind === "expression") {
    return FunctionIcon;
  }
  if (kind === "note") {
    return PencilEdit02Icon;
  }
  if (kind === "model_provider") {
    return Shield02Icon;
  }
  if (kind === "model_config") {
    return Plant01Icon;
  }
  if (kind === "tool_config") {
    return Plug01Icon;
  }
  if (kind === "seed") {
    return Plant01Icon;
  }
  return DiceFaces03Icon;
}

function getConfigSummary(config: NodeConfig | undefined): string {
  if (!config) {
    return "Open details for config";
  }

  if (config.kind === "sampler") {
    if (config.sampler_type === "category") {
      const count = config.values?.length ?? 0;
      return `${count} values`;
    }
    if (config.sampler_type === "subcategory") {
      if (config.subcategory_parent?.trim()) {
        return `Parent: ${config.subcategory_parent}`;
      }
      return "Select parent category";
    }
    if (config.sampler_type === "datetime") {
      const start = config.datetime_start?.trim() || "?";
      const end = config.datetime_end?.trim() || "?";
      return `${start} -> ${end}`;
    }
    if (config.sampler_type === "timedelta") {
      if (config.reference_column_name?.trim()) {
        return `Ref: ${config.reference_column_name}`;
      }
      return "Pick datetime reference";
    }
    if (
      config.sampler_type === "person" ||
      config.sampler_type === "person_from_faker"
    ) {
      const locale = config.person_locale?.trim() || "any locale";
      const city = config.person_city?.trim();
      if (city) {
        return `${locale} · ${city}`;
      }
      return locale;
    }
    return "Open details for config";
  }

  if (config.kind === "llm") {
    if (config.llm_type === "structured") {
      return "Structured output schema in details";
    }
    if (config.llm_type === "judge") {
      const scoreCount = config.scores?.length ?? 0;
      return `${scoreCount} scorers`;
    }
    if (config.tool_alias?.trim()) {
      return `Tool profile: ${config.tool_alias.trim()}`;
    }
    return "Prompt/system via linked input nodes";
  }

  if (config.kind === "tool_config") {
    const providerCount = config.mcp_providers.length;
    const allowCount = config.allow_tools?.filter((value) => value.trim()).length ?? 0;
    const providerLabel =
      providerCount === 1 ? "1 MCP server" : `${providerCount} MCP servers`;
    if (allowCount === 0) {
      return `${providerLabel} · all tools allowed`;
    }
    return `${providerLabel} · ${allowCount} allowed tools`;
  }

  if (config.kind === "validator") {
    const target = config.target_columns[0]?.trim();
    if (target) {
      return `Target: ${target}`;
    }
    return "Pick LLM code target";
  }

  if (config.kind === "seed") {
    const seedSourceType = config.seed_source_type ?? "hf";
    if (seedSourceType === "hf" && config.hf_repo_id.trim()) {
      return config.hf_repo_id.trim();
    }
    if (seedSourceType === "local" && config.local_file_name?.trim()) {
      return config.local_file_name.trim();
    }
    if (
      seedSourceType === "unstructured" &&
      config.unstructured_file_name?.trim()
    ) {
      return config.unstructured_file_name.trim();
    }
    if (config.hf_path.trim()) {
      return config.hf_path.trim();
    }
    if (seedSourceType === "hf") {
      return "Set HF dataset repo";
    }
    if (seedSourceType === "local") {
      return "Upload structured file";
    }
    return "Upload PDF/DOCX/TXT file";
  }

  if (config.kind === "markdown_note") {
    if (config.markdown.trim()) {
      return "Markdown preview";
    }
    return "Add markdown content";
  }

  return "Open details for config";
}

function renderNodeBody(
  config: NodeConfig | undefined,
  summary: string,
  updateConfig: (id: string, patch: Partial<NodeConfig>) => void,
): ReactElement {
  if (config?.kind === "markdown_note") {
    return <MarkdownPreview markdown={config.markdown} />;
  }

  if (config && isInlineConfig(config)) {
    const onUpdate = (patch: Partial<NodeConfig>) => updateConfig(config.id, patch);

    if (config.kind === "sampler") {
      return <InlineSampler config={config} onUpdate={onUpdate} />;
    }
    if (config.kind === "model_provider" || config.kind === "model_config") {
      return <InlineModel config={config} onUpdate={onUpdate} />;
    }
    if (config.kind === "llm") {
      return <InlineLlm config={config} onUpdate={onUpdate} />;
    }
    if (config.kind === "expression") {
      return <InlineExpression config={config} onUpdate={onUpdate} />;
    }
    if (config.kind === "seed") {
      return <InlineSeed config={config} onUpdate={onUpdate} />;
    }
  }

  if (config?.kind === "sampler" && config.sampler_type === "category") {
    return <InlineCategoryBadges values={config.values ?? []} />;
  }

  if (config?.kind === "tool_config") {
    const providerNames = config.mcp_providers
      .map((provider) => provider.name.trim())
      .filter(Boolean);
    return (
      <div className="space-y-2">
        <p className="text-xs text-muted-foreground">{summary}</p>
        {providerNames.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {providerNames.map((providerName) => (
              <Badge
                key={providerName}
                variant="secondary"
                className="corner-squircle font-mono text-[11px]"
              >
                {providerName}
              </Badge>
            ))}
          </div>
        )}
      </div>
    );
  }

  return <p className="text-xs text-muted-foreground">{summary}</p>;
}

function RecipeGraphNodeBase({
  id,
  data,
  selected,
}: NodeProps<RecipeGraphNodeType>): ReactElement {
  const meta = NODE_META[data.kind];
  const icon = resolveNodeIcon(data.kind, data.blockType);
  const layoutDirection = data.layoutDirection ?? "LR";
  const config = useRecipeStudioStore((state) => state.configs[id]);
  const openConfig = useRecipeStudioStore((state) => state.openConfig);
  const updateConfig = useRecipeStudioStore((state) => state.updateConfig);
  const llmAuxVisible = useRecipeStudioStore(
    (state) => state.llmAuxVisibility[id] ?? false,
  );
  const setLlmAuxVisibility = useRecipeStudioStore(
    (state) => state.setLlmAuxVisibility,
  );
  const updateNodeInternals = useUpdateNodeInternals();
  const executionLocked = Boolean(data.executionLocked);
  const runtimeState = data.runtimeState ?? "idle";

  useEffect(() => {
    updateNodeInternals(id);
  }, [id, layoutDirection, config, updateNodeInternals]);

  if (config?.kind === "markdown_note") {
    const rgb = hexToRgb(config.note_color ?? "#FDE68A");
    const alpha = parseNoteOpacity(config.note_opacity);
    const noteStyle = rgb
      ? {
          backgroundColor: `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${alpha})`,
          borderColor: `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${Math.min(1, Math.max(alpha + 0.15, 0.3))})`,
        }
      : undefined;

    return (
      <BaseNode
        className="corner-squircle relative w-full min-w-0 overflow-visible rounded-lg border-border/60 shadow-sm"
        style={noteStyle}
      >
        <NodeResizer
          isVisible={selected}
          minWidth={MIN_NODE_WIDTH}
          minHeight={80}
          maxWidth={MAX_NOTE_NODE_WIDTH}
          maxHeight={520}
          color="var(--primary)"
          lineClassName="!border-transparent !shadow-none"
          lineStyle={{ opacity: 0 }}
          handleClassName="!h-3 !w-3 !border-transparent !bg-transparent"
          handleStyle={{ opacity: 0 }}
        />
        <BaseNodeContent className="px-3 py-2">
          <MarkdownPreview markdown={config.markdown} plain={true} />
        </BaseNodeContent>
      </BaseNode>
    );
  }

  const showDataHandles =
    data.kind === "llm" ||
    data.kind === "validator" ||
    data.kind === "expression" ||
    data.kind === "sampler" ||
    data.kind === "seed";
  const showSemanticIn = data.kind === "model_config" || data.kind === "validator";
  const showSemanticOut =
    data.kind === "model_config" ||
    data.kind === "model_provider" ||
    data.kind === "tool_config" ||
    data.kind === "validator";
  const summary = getConfigSummary(config);
  const nodeBody = renderNodeBody(config, summary, updateConfig);
  const canShowLlmAux =
    config?.kind === "llm" &&
    (Boolean(config.prompt.trim()) ||
      Boolean(config.system_prompt.trim()) ||
      Boolean((config.scores?.length ?? 0) > 0));
  const iconTone =
    config?.kind === "sampler" &&
    (config.sampler_type === "person" ||
      config.sampler_type === "person_from_faker")
      ? USER_NODE_TONE
      : meta.tone;
  const runtimeNodeTone =
    runtimeState === "running"
      ? "border-primary/70 ring-2 ring-primary/20 shadow-md"
      : runtimeState === "done"
        ? "border-emerald-500/60 ring-1 ring-emerald-500/20"
        : "";

  return (
    <BaseNode
      className={cn(
        "corner-squircle relative w-full min-w-0 overflow-visible rounded-lg border-border/60 shadow-sm",
        runtimeNodeTone,
      )}
    >
      {runtimeState === "running" && config?.kind === "llm" && (
        <div className="pointer-events-none absolute -top-7 right-2 z-20">
          <span
            className="block size-6 animate-spin rounded-full border-[3px] border-primary/90 border-t-transparent bg-background"
            aria-label="Running"
          />
        </div>
      )}
      <NodeResizer
        isVisible={selected}
        minWidth={MIN_NODE_WIDTH}
        minHeight={120}
        maxWidth={MAX_NODE_WIDTH}
        maxHeight={520}
        color="var(--primary)"
        lineClassName="!border-transparent !shadow-none"
        lineStyle={{ opacity: 0 }}
        handleClassName="!h-3 !w-3 !border-transparent !bg-transparent"
        handleStyle={{ opacity: 0 }}
      />
      <BaseNodeHeader className="border-b border-border/50 px-3 py-2">
        <div className="flex min-w-0 items-center gap-2">
          <div
            className={cn(
              "corner-squircle flex size-7 items-center justify-center rounded-md border",
              iconTone,
            )}
          >
            <HugeiconsIcon icon={icon} className="size-3.5" />
          </div>
          <div className="min-w-0">
            <BaseNodeHeaderTitle className="truncate text-sm">
              {data.name}
            </BaseNodeHeaderTitle>
            <p className="truncate text-[11px] text-muted-foreground">
              {data.subtype} · {data.title}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-1">
          {canShowLlmAux && (
            <Button
              type="button"
              size="xs"
              variant="ghost"
              className="nodrag"
              disabled={executionLocked}
              onClick={(event) => {
                event.preventDefault();
                event.stopPropagation();
                setLlmAuxVisibility(id, !llmAuxVisible);
              }}
            >
              {llmAuxVisible ? "Hide inputs" : "Show inputs"}
            </Button>
          )}
          <Button
            type="button"
            size="xs"
            variant="ghost"
            className="nodrag"
            disabled={executionLocked}
            onClick={(event) => {
              event.preventDefault();
              event.stopPropagation();
              openConfig(id);
            }}
          >
            Configure
          </Button>
        </div>
      </BaseNodeHeader>

      <BaseNodeContent
        className={cn(
          "gap-2 px-3 py-2",
          executionLocked && "pointer-events-none opacity-85",
        )}
      >
        {nodeBody}
      </BaseNodeContent>

      {showDataHandles && (
        <>
          <LabeledHandle
            id={HANDLE_IDS.dataIn}
            title="Data input"
            type="target"
            position={Position.Left}
            className="absolute inset-0 pointer-events-none"
            labelClassName="sr-only"
            handleClassName={NODE_HANDLE_CLASS}
          />
          <LabeledHandle
            id={HANDLE_IDS.dataOutLeft}
            title="Data output"
            type="source"
            position={Position.Left}
            className="absolute inset-0 pointer-events-none opacity-0"
            labelClassName="sr-only"
            handleClassName={NODE_HANDLE_CLASS}
          />
          <LabeledHandle
            id={HANDLE_IDS.dataInTop}
            title="Data input"
            type="target"
            position={Position.Top}
            className="absolute inset-0 pointer-events-none"
            labelClassName="sr-only"
            handleClassName={NODE_HANDLE_CLASS}
          />
          <LabeledHandle
            id={HANDLE_IDS.dataOutTop}
            title="Data output"
            type="source"
            position={Position.Top}
            className="absolute inset-0 pointer-events-none opacity-0"
            labelClassName="sr-only"
            handleClassName={NODE_HANDLE_CLASS}
          />
          <LabeledHandle
            id={HANDLE_IDS.dataOut}
            title="Data output"
            type="source"
            position={Position.Right}
            className="absolute inset-0 pointer-events-none"
            labelClassName="sr-only"
            handleClassName={NODE_HANDLE_CLASS}
          />
          <LabeledHandle
            id={HANDLE_IDS.dataInRight}
            title="Data input"
            type="target"
            position={Position.Right}
            className="absolute inset-0 pointer-events-none opacity-0"
            labelClassName="sr-only"
            handleClassName={NODE_HANDLE_CLASS}
          />
          <LabeledHandle
            id={HANDLE_IDS.dataOutBottom}
            title="Data output"
            type="source"
            position={Position.Bottom}
            className="absolute inset-0 pointer-events-none"
            labelClassName="sr-only"
            handleClassName={NODE_HANDLE_CLASS}
          />
          <LabeledHandle
            id={HANDLE_IDS.dataInBottom}
            title="Data input"
            type="target"
            position={Position.Bottom}
            className="absolute inset-0 pointer-events-none opacity-0"
            labelClassName="sr-only"
            handleClassName={NODE_HANDLE_CLASS}
          />
        </>
      )}

      {showSemanticIn && (
        <>
          <LabeledHandle
            id={HANDLE_IDS.semanticIn}
            title="Semantic input"
            type="target"
            position={Position.Left}
            className="absolute inset-0 pointer-events-none"
            labelClassName="sr-only"
            handleClassName={NODE_HANDLE_CLASS}
          />
          <LabeledHandle
            id={HANDLE_IDS.semanticInTop}
            title="Semantic input"
            type="target"
            position={Position.Top}
            className="absolute inset-0 pointer-events-none"
            labelClassName="sr-only"
            handleClassName={NODE_HANDLE_CLASS}
          />
        </>
      )}

      {showSemanticOut && (
        <>
          <LabeledHandle
            id={HANDLE_IDS.semanticOut}
            title="Semantic output"
            type="source"
            position={Position.Right}
            className="absolute inset-0 pointer-events-none"
            labelClassName="sr-only"
            handleClassName={NODE_HANDLE_CLASS}
          />
          <LabeledHandle
            id={HANDLE_IDS.semanticOutBottom}
            title="Semantic output"
            type="source"
            position={Position.Bottom}
            className="absolute inset-0 pointer-events-none"
            labelClassName="sr-only"
            handleClassName={NODE_HANDLE_CLASS}
          />
        </>
      )}
    </BaseNode>
  );
}

export const RecipeNode = memo(RecipeGraphNodeBase);
