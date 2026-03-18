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
  CheckmarkCircle01Icon,
  InformationCircleIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { EXPORT_METHODS, type ExportMethod } from "../constants";

interface MethodPickerProps {
  value: ExportMethod | null;
  onChange: (v: ExportMethod) => void;
  /** Methods that should be shown but disabled (greyed out, not clickable). */
  disabledMethods?: ExportMethod[];
  /** Optional reason shown in a tooltip on disabled methods. */
  disabledReason?: string;
}

export function MethodPicker({ value, onChange, disabledMethods = [], disabledReason }: MethodPickerProps) {
  return (
    <div data-tour="export-method" className="flex flex-col gap-3">
      <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
        Export Method
        <Tooltip>
          <TooltipTrigger asChild={true}>
            <button
              type="button"
              className="text-foreground/70 hover:text-foreground"
            >
              <HugeiconsIcon icon={InformationCircleIcon} className="size-3" />
            </button>
          </TooltipTrigger>
          <TooltipContent>
            How your model is packaged for deployment.{" "}
            <a
              href="https://unsloth.ai/docs/basics/inference-and-deployment"
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary underline"
            >
              Read more
            </a>
          </TooltipContent>
        </Tooltip>
      </span>
      <div className="grid grid-cols-3 gap-3">
        {EXPORT_METHODS.map((m) => {
          const selected = value === m.value;
          const isDisabled = disabledMethods.includes(m.value);

          const card = (
            <button
              key={m.value}
              type="button"
              disabled={isDisabled}
              onClick={() => !isDisabled && onChange(m.value)}
              className={cn(
                "flex items-start gap-3 rounded-xl p-4 text-left ring-1 transition-all",
                isDisabled
                  ? "ring-border opacity-40 cursor-not-allowed"
                  : selected
                    ? "ring-2 ring-primary bg-primary/5"
                    : "ring-border hover:-translate-y-0.5 hover:shadow-sm",
              )}
            >
              <div
                className={cn(
                  "mt-0.5 flex size-5 shrink-0 items-center justify-center rounded-full border-2 transition-colors",
                  selected
                    ? "border-primary bg-primary"
                    : "border-muted-foreground/30",
                )}
              >
                {selected && (
                  <HugeiconsIcon
                    icon={CheckmarkCircle01Icon}
                    className="size-3 text-primary-foreground"
                  />
                )}
              </div>
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
                      className="text-[10px] px-1.5 py-0"
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
