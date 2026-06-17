// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  useDownloadTransportCapabilities,
  useTransportMode,
} from "@/features/hub/download-manager";
import { cn } from "@/lib/utils";
import { useEffect } from "react";

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
  const { capabilities } = useDownloadTransportCapabilities();
  const xetUnavailable = capabilities?.xet.available === false;

  useEffect(() => {
    if (mode === "xet" && xetUnavailable) {
      setMode("http");
    }
  }, [mode, setMode, xetUnavailable]);

  return (
    <fieldset
      aria-label="Download transport"
      className="hub-tag-soft m-0 inline-flex h-[26px] min-w-0 items-center gap-0.5 rounded-full border-0 p-0.5 text-[11px]"
    >
      {OPTIONS.map((opt) => {
        const active = mode === opt.value;
        const disabled = opt.value === "xet" && xetUnavailable;
        const hint =
          disabled && capabilities?.xet.reason
            ? capabilities.xet.reason
            : opt.hint;
        return (
          <Tooltip key={opt.value}>
            <TooltipTrigger asChild={true}>
              <button
                type="button"
                aria-disabled={disabled || undefined}
                aria-pressed={active}
                onClick={() => {
                  if (!disabled) setMode(opt.value);
                }}
                className={cn(
                  "inline-flex h-[22px] items-center justify-center rounded-full px-2 font-medium tracking-tight transition-colors",
                  disabled
                    ? "cursor-not-allowed text-muted-foreground/45"
                    : active
                      ? "hub-tab-toggle-pill cursor-pointer text-foreground"
                      : "cursor-pointer text-muted-foreground hover:text-foreground/80",
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
              {hint}
            </TooltipContent>
          </Tooltip>
        );
      })}
    </fieldset>
  );
}
