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
import {
  Handle,
  NodeResizer,
  Position,
  useUpdateNodeInternals,
  type NodeProps,
} from "@xyflow/react";
import { memo, type ReactElement, useEffect } from "react";
import { useCanvasLabStore } from "../stores/canvas-lab";
import type {
  CanvasNode as CanvasNodeType,
  LlmType,
  NodeConfig,
  SamplerType,
} from "../types";
import { getLlmJudgeScoreHandleId, HANDLE_IDS } from "../utils/handles";
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
    return "Prompt/system via linked input nodes";
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

type LlmInputHandleItem = {
  id: string;
  label: string;
};

function getLlmInputHandleItems(config: NodeConfig | undefined): LlmInputHandleItem[] {
  if (!(config && config.kind === "llm")) {
    return [];
  }
  const items: LlmInputHandleItem[] = [];
  if (config.system_prompt.trim()) {
    items.push({ id: HANDLE_IDS.llmSystemIn, label: "System" });
  }
  if (config.prompt.trim()) {
    items.push({ id: HANDLE_IDS.llmPromptIn, label: "Prompt" });
  }
  if (config.llm_type === "judge") {
    (config.scores ?? []).forEach((score, index) => {
      items.push({
        id: getLlmJudgeScoreHandleId(index),
        label: score.name.trim() || `Score ${index + 1}`,
      });
    });
  }
  return items;
}

type LlmInputHandlesProps = {
  items: LlmInputHandleItem[];
  isTopBottom: boolean;
};

function LlmInputHandles({ items, isTopBottom }: LlmInputHandlesProps): ReactElement | null {
  if (items.length === 0) {
    return null;
  }

  if (isTopBottom) {
    return (
      <div className="flex flex-wrap gap-2 pb-1">
        {items.map((item) => (
          <div
            key={item.id}
            className="pointer-events-none relative flex min-w-[80px] flex-1 justify-center pt-2"
          >
            <Handle
              id={item.id}
              type="target"
              position={Position.Top}
              className="pointer-events-auto !size-2 !border-border !bg-background"
              style={{ left: "50%", top: 0, transform: "translate(-50%, -50%)" }}
            />
            <span className="text-[10px] text-muted-foreground">{item.label}</span>
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="space-y-1 pb-1">
      {items.map((item) => (
        <div key={item.id} className="pointer-events-none relative pl-3">
          <Handle
            id={item.id}
            type="target"
            position={Position.Left}
            className="pointer-events-auto !size-2 !border-border !bg-background"
            style={{ left: -3, top: "50%", transform: "translate(-50%, -50%)" }}
          />
          <span className="text-[10px] text-muted-foreground">{item.label}</span>
        </div>
      ))}
    </div>
  );
}

function CanvasNodeBase({
  id,
  data,
  selected,
}: NodeProps<CanvasNodeType>): ReactElement {
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
  const llmInputHandles = getLlmInputHandleItems(config);

  return (
    <BaseNode className="corner-squircle relative min-w-[260px] overflow-visible rounded-lg border-border/60 shadow-sm">
      <NodeResizer
        isVisible={selected}
        minWidth={260}
        minHeight={120}
        maxWidth={760}
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
        <LlmInputHandles items={llmInputHandles} isTopBottom={isTopBottom} />
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
