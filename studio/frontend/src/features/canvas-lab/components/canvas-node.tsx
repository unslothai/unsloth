import { cn } from "@/lib/utils";
import {
  CodeIcon,
  Database02Icon,
  SparklesIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { NodeProps } from "@xyflow/react";
import { Handle, Position, useUpdateNodeInternals } from "@xyflow/react";
import { type ReactElement, memo, useEffect } from "react";
import type { CanvasNode as CanvasNodeType } from "../types";

const NODE_META = {
  sampler: {
    icon: Database02Icon,
    tone: "bg-emerald-50 text-emerald-600 border-emerald-100",
  },
  llm: {
    icon: SparklesIcon,
    tone: "bg-purple-50 text-purple-600 border-purple-100",
  },
  expression: {
    icon: CodeIcon,
    tone: "bg-sky-50 text-sky-600 border-sky-100",
  },
} as const;

function CanvasNodeBase({
  id,
  data,
  selected,
}: NodeProps<CanvasNodeType>): ReactElement {
  const meta = NODE_META[data.kind];
  const layoutDirection = data.layoutDirection ?? "LR";
  const isHorizontal = layoutDirection === "LR";
  const updateNodeInternals = useUpdateNodeInternals();

  useEffect(() => {
    updateNodeInternals(id);
  }, [id, layoutDirection, updateNodeInternals]);

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
          <HugeiconsIcon icon={meta.icon} className="size-4" />
        </div>
        <div>
          <p className="text-sm font-semibold text-foreground">{data.title}</p>
          <p className="text-[11px] text-muted-foreground">
            {data.subtype} · {data.name}
          </p>
        </div>
      </div>
      <Handle
        type="target"
        position={isHorizontal ? Position.Left : Position.Top}
        className="size-2 border border-border bg-white"
      />
      <Handle
        type="source"
        position={isHorizontal ? Position.Right : Position.Bottom}
        className="size-2 border border-border bg-white"
      />
    </div>
  );
}

export const CanvasNode = memo(CanvasNodeBase);
