// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import {
  CheckmarkCircle01Icon,
  InformationCircleIcon,
  LayersIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { QUANT_OPTIONS } from "../constants";

interface QuantPickerProps {
  value: string[];
  onChange: (v: string[]) => void;
}

export function QuantPicker({ value, onChange }: QuantPickerProps) {
  const toggle = (qv: string) => {
    onChange(
      value.includes(qv) ? value.filter((q) => q !== qv) : [...value, qv],
    );
  };

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-2">
        <HugeiconsIcon
          icon={LayersIcon}
          className="size-4 text-muted-foreground"
        />
        <span className="text-xs font-medium text-muted-foreground">
          量化等级
        </span>
        <Tooltip>
          <TooltipTrigger asChild={true}>
            <button
              type="button"
              className="text-foreground/70 hover:text-foreground"
            >
              <HugeiconsIcon icon={InformationCircleIcon} className="size-3" />
            </button>
          </TooltipTrigger>
          <TooltipContent className="max-w-xs">
            更低量化（Q2、Q3）文件更小但质量下降，Q4-Q5 通常更均衡。{" "}
            <a
              href="https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf"
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary underline"
            >
              了解更多
            </a>
          </TooltipContent>
        </Tooltip>
        <span className="text-[11px] text-muted-foreground/70">
          - 可多选
        </span>
      </div>
      <div className="flex flex-wrap gap-2 py-1 pl-1">
        {QUANT_OPTIONS.map((q) => {
          const active = value.includes(q.value);
          return (
            <button
              key={q.value}
              type="button"
              onClick={() => toggle(q.value)}
              className={cn(
                "inline-flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-medium ring-1 transition-all",
                active
                  ? "ring-primary bg-primary/10 text-foreground"
                  : "ring-border text-muted-foreground hover:text-foreground hover:ring-foreground/20",
              )}
            >
              {active && (
                <HugeiconsIcon
                  icon={CheckmarkCircle01Icon}
                  className="size-3 text-primary"
                />
              )}
              {q.label}
              <span className="text-[10px] opacity-60">{q.size}</span>
              {q.recommended && !active && (
                <span className="rounded-full bg-emerald-100 px-1.5 py-0 text-[9px] font-semibold text-emerald-700 dark:bg-emerald-900 dark:text-emerald-300">
                  推荐
                </span>
              )}
            </button>
          );
        })}
      </div>
      {value.length > 0 && (
        <div className="flex items-center gap-3">
          <span className="text-[11px] text-muted-foreground">
            已选择 {value.length} 项
          </span>
          <button
            type="button"
            onClick={() => onChange([])}
            className="text-[11px] text-muted-foreground/70 hover:text-foreground transition-colors"
          >
            清空
          </button>
        </div>
      )}
    </div>
  );
}
