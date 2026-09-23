// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Tick02Icon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";
import { Copy01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { MouseEvent } from "react";
import { useCopyFeedback } from "../hooks/use-copy-feedback";

/** Copies the on-disk path straight to the clipboard, no dialog. */
export function PathInfoButton({
  path,
  className,
}: {
  path: string;
  className?: string;
}) {
  const { copied, copy } = useCopyFeedback();

  const handleCopy = async (event: MouseEvent<HTMLButtonElement>) => {
    event.preventDefault();
    event.stopPropagation();
    await copy(path);
  };

  return (
    <Tooltip>
      <TooltipTrigger asChild={true}>
        <button
          type="button"
          aria-label="Copy on-device path"
          onClick={handleCopy}
          className={cn(
            "inline-flex size-7 shrink-0 cursor-pointer items-center justify-center rounded-full text-muted-foreground opacity-0 transition-[opacity,background-color,color] duration-150 hover:bg-muted hover:text-foreground focus-visible:opacity-100 group-hover/dl:opacity-100",
            className,
          )}
        >
          <HugeiconsIcon
            icon={copied ? Tick02Icon : Copy01Icon}
            strokeWidth={1.75}
            className="size-4"
          />
        </button>
      </TooltipTrigger>
      <TooltipContent side="bottom" className="tooltip-compact">
        {copied ? "Copied" : "Copy path"}
      </TooltipContent>
    </Tooltip>
  );
}
