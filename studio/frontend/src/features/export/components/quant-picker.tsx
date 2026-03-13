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
          Quantization Levels
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
            Lower quantization (Q2, Q3) = smaller files but reduced quality.
            Q4–Q5 is a good balance.{" "}
            <a
              href="https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf"
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary underline"
            >
              Read more
            </a>
          </TooltipContent>
        </Tooltip>
        <span className="text-[11px] text-muted-foreground/70">
          — select one or more
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
                  rec
                </span>
              )}
            </button>
          );
        })}
      </div>
      {value.length > 0 && (
        <div className="flex items-center gap-3">
          <span className="text-[11px] text-muted-foreground">
            {value.length} selected
          </span>
          <button
            type="button"
            onClick={() => onChange([])}
            className="text-[11px] text-muted-foreground/70 hover:text-foreground transition-colors"
          >
            Clear all
          </button>
        </div>
      )}
    </div>
  );
}
