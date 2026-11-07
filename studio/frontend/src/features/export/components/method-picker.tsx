// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import {
  InformationCircleIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { EXPORT_METHODS, type ExportMethod } from "../constants";

interface MethodPickerProps {
  value: ExportMethod | null;
  onChange: (v: ExportMethod) => void;
  disabledMethods?: ExportMethod[];
  disabledReason?: string;
}

export function MethodPicker({
  value,
  onChange,
  disabledMethods = [],
  disabledReason,
}: MethodPickerProps) {
  return (
    <div data-tour="export-method" className="flex flex-col gap-3">
      <div className="grid grid-cols-1 gap-3 @xl/train-section:grid-cols-3">
        {EXPORT_METHODS.map((m) => {
          const selected = value === m.value;
          const isDisabled = disabledMethods.includes(m.value);

          const card = (
            <button
              key={m.value}
              type="button"
              disabled={isDisabled}
              aria-pressed={selected}
              onClick={() => !isDisabled && onChange(m.value)}
              className={cn(
                "flex items-start rounded-[18px] p-4 text-left ring-1 transition-colors",
                isDisabled
                  ? "ring-border opacity-40 cursor-not-allowed"
                  : selected
                    ? "ring-transparent bg-wash-5"
                    : "ring-border hover:bg-muted/40",
              )}
            >
              <div className="flex flex-col gap-1">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium">{m.title}</span>
                  <Tooltip>
                    <TooltipTrigger asChild={true}>
                      <button
                        type="button"
                        className="shrink-0 text-foreground/50 hover:text-foreground cursor-help"
                        onClick={(e) => e.stopPropagation()}
                        aria-label={`${m.title} info`}
                      >
                        <HugeiconsIcon
                          icon={InformationCircleIcon}
                          className="size-3"
                        />
                      </button>
                    </TooltipTrigger>
                    <TooltipContent className="max-w-xs">
                      {m.tooltip}{" "}
                      <a
                        href={
                          m.value === "gguf"
                            ? "https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf"
                            : "https://unsloth.ai/docs/basics/inference-and-deployment"
                        }
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-primary underline"
                      >
                        Read more
                      </a>
                    </TooltipContent>
                  </Tooltip>
                  {m.badge && (
                    <Badge
                      variant="secondary"
                      className="text-ui-10 px-1.5 py-0"
                    >
                      {m.badge}
                    </Badge>
                  )}
                </div>
                <span className="text-xs text-muted-foreground">
                  {m.description}
                </span>
              </div>
            </button>
          );

          if (isDisabled && disabledReason) {
            return (
              <Tooltip key={m.value}>
                <TooltipTrigger asChild={true}>{card}</TooltipTrigger>
                <TooltipContent>{disabledReason}</TooltipContent>
              </Tooltip>
            );
          }

          return card;
        })}
      </div>
    </div>
  );
}
