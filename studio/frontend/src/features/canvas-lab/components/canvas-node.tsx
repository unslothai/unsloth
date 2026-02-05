import { Button } from "@/components/ui/button";
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
  Parabola02Icon,
  PencilEdit02Icon,
  Plant01Icon,
  Shield02Icon,
  Tag01Icon,
  TagsIcon,
  UserAccountIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Position, useUpdateNodeInternals, type NodeProps } from "@xyflow/react";
import { memo, type ReactElement, useEffect } from "react";
import { useCanvasLabStore } from "../stores/canvas-lab";
import type {
  CanvasNode as CanvasNodeType,
  LlmType,
  NodeConfig,
  SamplerType,
} from "../types";
import { HANDLE_IDS } from "../utils/handles";
import { InlineExpression } from "./inline/inline-expression";
import { InlineLlm } from "./inline/inline-llm";
import { InlineModel } from "./inline/inline-model";
import { isInlineConfig } from "./inline/inline-policy";
import { InlineSampler } from "./inline/inline-sampler";
import {
  BaseNode,
  BaseNodeContent,
  BaseNodeHeader,
  BaseNodeHeaderTitle,
} from "./rf-ui/base-node";
import { LabeledHandle } from "./rf-ui/labeled-handle";

type IconType = typeof CodeIcon;

const NODE_META = {
  sampler: {
    tone: "bg-emerald-50 text-emerald-600 border-emerald-100",
  },
  llm: {
    tone: "bg-sky-50 text-sky-600 border-sky-100",
  },
  expression: {
    tone: "bg-indigo-50 text-indigo-600 border-indigo-100",
  },
  model_provider: {
    tone: "bg-amber-50 text-amber-600 border-amber-100",
  },
  model_config: {
    tone: "bg-orange-50 text-orange-600 border-orange-100",
  },
} as const;

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
  kind: CanvasNodeType["data"]["kind"],
  blockType: CanvasNodeType["data"]["blockType"],
): IconType {
  if (kind === "sampler" && blockType in SAMPLER_ICONS) {
    return SAMPLER_ICONS[blockType as SamplerType];
  }
  if (kind === "llm" && blockType in LLM_ICONS) {
    return LLM_ICONS[blockType as LlmType];
  }
  if (kind === "expression") {
    return FunctionIcon;
  }
  if (kind === "model_provider") {
    return Shield02Icon;
  }
  if (kind === "model_config") {
    return Plant01Icon;
  }
  return DiceFaces03Icon;
}

function toSingleLine(value: string | undefined): string {
  if (!value) {
    return "";
  }
  const normalized = value.replace(/\s+/g, " ").trim();
  if (!normalized) {
    return "";
  }
  if (normalized.length <= 96) {
    return normalized;
  }
  return `${normalized.slice(0, 93)}...`;
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
    const prompt = toSingleLine(config.prompt);
    if (config.llm_type === "structured") {
      if (prompt) {
        return `Prompt: ${prompt}`;
      }
      return "Structured output schema in details";
    }
    if (config.llm_type === "judge") {
      const scoreCount = config.scores?.length ?? 0;
      if (prompt) {
        return `${scoreCount} scores · ${prompt}`;
      }
      return `${scoreCount} scores`;
    }
    return "Open details for config";
  }

  return "Open details for config";
}

function renderInlineEditor(
  config: NodeConfig | undefined,
  updateConfig: (id: string, patch: Partial<NodeConfig>) => void,
): ReactElement | null {
  if (!config || !isInlineConfig(config)) {
    return null;
  }

  if (config.kind === "sampler") {
    return (
      <InlineSampler
        config={config}
        onUpdate={(patch) => updateConfig(config.id, patch)}
      />
    );
  }

  if (config.kind === "model_provider" || config.kind === "model_config") {
    return (
      <InlineModel
        config={config}
        onUpdate={(patch) => updateConfig(config.id, patch)}
      />
    );
  }

  if (config.kind === "llm") {
    return (
      <InlineLlm config={config} onUpdate={(patch) => updateConfig(config.id, patch)} />
    );
  }

  if (config.kind === "expression") {
    return (
      <InlineExpression
        config={config}
        onUpdate={(patch) => updateConfig(config.id, patch)}
      />
    );
  }

  return null;
}

function CanvasNodeBase({ id, data }: NodeProps<CanvasNodeType>): ReactElement {
  const meta = NODE_META[data.kind];
  const icon = resolveNodeIcon(data.kind, data.blockType);
  const layoutDirection = data.layoutDirection ?? "LR";
  const config = useCanvasLabStore((state) => state.configs[id]);
  const openConfig = useCanvasLabStore((state) => state.openConfig);
  const updateConfig = useCanvasLabStore((state) => state.updateConfig);
  const updateNodeInternals = useUpdateNodeInternals();

  useEffect(() => {
    updateNodeInternals(id);
  }, [id, layoutDirection, config, updateNodeInternals]);

  const showDataHandles =
    data.kind === "llm" || data.kind === "expression" || data.kind === "sampler";
  const showSemanticIn = data.kind === "llm" || data.kind === "model_config";
  const showSemanticOut = data.kind === "model_config" || data.kind === "model_provider";
  const isTopBottom = layoutDirection === "TB";

  const dataInPosition = isTopBottom ? Position.Top : Position.Left;
  const dataOutPosition = isTopBottom ? Position.Bottom : Position.Right;
  const semanticInPosition = isTopBottom ? Position.Left : Position.Top;
  const semanticOutPosition = isTopBottom ? Position.Right : Position.Bottom;

  const inlineEditor = renderInlineEditor(config, updateConfig);
  const summary = getConfigSummary(config);

  return (
    <BaseNode className="relative min-w-[260px] overflow-visible rounded-xl border-border/60 shadow-sm">
      <BaseNodeHeader className="border-b border-border/50 px-3 py-2">
        <div className="flex min-w-0 items-center gap-2">
          <div
            className={cn(
              "flex size-7 items-center justify-center rounded-md border",
              meta.tone,
            )}
          >
            <HugeiconsIcon icon={icon} className="size-3.5" />
          </div>
          <div className="min-w-0">
            <BaseNodeHeaderTitle className="truncate text-sm">
              {data.title}
            </BaseNodeHeaderTitle>
            <p className="truncate text-[11px] text-muted-foreground">
              {data.subtype} · {data.name}
            </p>
          </div>
        </div>
        <Button
          type="button"
          size="xs"
          variant="ghost"
          className="nodrag"
          onClick={(event) => {
            event.preventDefault();
            event.stopPropagation();
            openConfig(id);
          }}
        >
          Details
        </Button>
      </BaseNodeHeader>

      <BaseNodeContent className="gap-2 px-3 py-2">
        {inlineEditor ? (
          inlineEditor
        ) : (
          <p className="text-xs text-muted-foreground">{summary}</p>
        )}
      </BaseNodeContent>

      {showDataHandles && (
        <>
          <LabeledHandle
            id={HANDLE_IDS.dataIn}
            title="Data input"
            type="target"
            position={dataInPosition}
            className="absolute inset-0 pointer-events-none"
            labelClassName="sr-only"
            handleClassName="pointer-events-auto !size-2 !border-border !bg-background"
          />
          <LabeledHandle
            id={HANDLE_IDS.dataOut}
            title="Data output"
            type="source"
            position={dataOutPosition}
            className="absolute inset-0 pointer-events-none"
            labelClassName="sr-only"
            handleClassName="pointer-events-auto !size-2 !border-border !bg-background"
          />
        </>
      )}

      {showSemanticIn && (
        <LabeledHandle
          id={HANDLE_IDS.semanticIn}
          title="Semantic input"
          type="target"
          position={semanticInPosition}
          className="absolute inset-0 pointer-events-none"
          labelClassName="sr-only"
          handleClassName="pointer-events-auto !size-2 !border-border !bg-background"
        />
      )}

      {showSemanticOut && (
        <LabeledHandle
          id={HANDLE_IDS.semanticOut}
          title="Semantic output"
          type="source"
          position={semanticOutPosition}
          className="absolute inset-0 pointer-events-none"
          labelClassName="sr-only"
          handleClassName="pointer-events-auto !size-2 !border-border !bg-background"
        />
      )}
    </BaseNode>
  );
}

export const CanvasNode = memo(CanvasNodeBase);
