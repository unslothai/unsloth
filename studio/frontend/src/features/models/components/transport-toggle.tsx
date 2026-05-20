// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { useTransportMode } from "../lib/transport-preference";

const OPTIONS: { value: "http" | "xet"; label: string; hint: string }[] = [
  {
    value: "http",
    label: "HTTP",
    hint: "Resumes from where it stopped if you cancel.",
  },
  {
    value: "xet",
    label: "Xet",
    hint: "Usually faster on a fresh download, but starts over if you cancel.",
  },
];

export function TransportToggle() {
  const [mode, setMode] = useTransportMode();
  return (
    <div
      role="group"
      aria-label="Download transport"
      className="tag-soft inline-flex h-[26px] items-center gap-0.5 rounded-[11px] p-0.5 text-[11px]"
    >
      {OPTIONS.map((opt) => {
        const active = mode === opt.value;
        return (
          <Tooltip key={opt.value}>
            <TooltipTrigger asChild>
              <button
                type="button"
                onClick={() => setMode(opt.value)}
                aria-pressed={active}
                className={cn(
                  "inline-flex h-[22px] cursor-pointer items-center justify-center rounded-[8px] px-2 font-medium tracking-tight transition-colors",
                  active
                    ? "bg-foreground/[0.08] text-foreground dark:bg-white/[0.08]"
                    : "text-muted-foreground hover:text-foreground/80",
                )}
              >
                {opt.label}
              </button>
            </TooltipTrigger>
            <TooltipContent
              side="bottom"
              sideOffset={6}
              className="tooltip-compact"
            >
              {opt.hint}
            </TooltipContent>
          </Tooltip>
        );
      })}
    </div>
  );
}
