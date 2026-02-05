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
import type { NodeProps } from "@xyflow/react";
import { Handle, Position, useUpdateNodeInternals } from "@xyflow/react";
import { type ReactElement, memo, useEffect } from "react";
import type {
  CanvasNode as CanvasNodeType,
  LlmType,
  SamplerType,
} from "../types";
import { HANDLE_IDS } from "../utils/handles";

type IconType = typeof CodeIcon;

const NODE_META = {
  sampler: {
    tone: "bg-emerald-50 text-emerald-600 border-emerald-100",
  },
  llm: {
    tone: "bg-purple-50 text-purple-600 border-purple-100",
  },
  expression: {
    tone: "bg-sky-50 text-sky-600 border-sky-100",
  },
  model_provider: {
    tone: "bg-amber-50 text-amber-600 border-amber-100",
  },
  model_config: {
    tone: "bg-indigo-50 text-indigo-600 border-indigo-100",
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

function CanvasNodeBase({
  id,
  data,
  selected,
}: NodeProps<CanvasNodeType>): ReactElement {
  const meta = NODE_META[data.kind];
  const icon = resolveNodeIcon(data.kind, data.blockType);
  const layoutDirection = data.layoutDirection ?? "LR";
  const updateNodeInternals = useUpdateNodeInternals();

  useEffect(() => {
    updateNodeInternals(id);
  }, [id, layoutDirection, updateNodeInternals]);

  const showDataHandles =
    data.kind === "llm" ||
    data.kind === "expression" ||
    (data.kind === "sampler" &&
      data.blockType !== "model_provider" &&
      data.blockType !== "model_config");
  const showSemanticIn =
    data.kind === "llm" ||
    data.kind === "model_config";
  const showSemanticOut =
    data.kind === "llm" ||
    data.kind === "model_config" ||
    data.kind === "model_provider";

  return (
    <div
      className={cn(
        "rounded-2xl border bg-white px-4 py-3 shadow-sm min-w-[180px]",
        selected
          ? "border-foreground/40 ring-1 ring-foreground/10"
          : "border-border/60",
      )}
    >
      <div className="flex items-center gap-3">
        <div
          className={cn(
            "flex size-9 items-center justify-center rounded-xl border",
            meta.tone,
          )}
        >
          <HugeiconsIcon icon={icon} className="size-4" />
        </div>
        <div>
          <p className="text-sm font-semibold text-foreground">{data.title}</p>
          <p className="text-[11px] text-muted-foreground">
            {data.subtype} · {data.name}
          </p>
        </div>
      </div>
      {showDataHandles && (
        <>
          <Handle
            id={HANDLE_IDS.dataIn}
            type="target"
            position={Position.Left}
            className="size-2 border border-border bg-white"
          />
          <Handle
            id={HANDLE_IDS.dataOut}
            type="source"
            position={Position.Right}
            className="size-2 border border-border bg-white"
          />
        </>
      )}
      {showSemanticIn && (
        <Handle
          id={HANDLE_IDS.semanticIn}
          type="target"
          position={Position.Top}
          className="size-2 border border-border bg-white"
        />
      )}
      {showSemanticOut && (
        <Handle
          id={HANDLE_IDS.semanticOut}
          type="source"
          position={Position.Bottom}
          className="size-2 border border-border bg-white"
        />
      )}
    </div>
  );
}

export const CanvasNode = memo(CanvasNodeBase);
