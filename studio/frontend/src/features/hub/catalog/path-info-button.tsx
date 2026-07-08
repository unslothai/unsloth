// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { Copy01Icon, FolderSearchIcon } from "@hugeicons/core-free-icons";
import { Tick02Icon } from "@/lib/tick-icon";
import { HugeiconsIcon } from "@hugeicons/react";
import type { MouseEvent } from "react";
import { useState } from "react";
import { useCopyFeedback } from "../hooks/use-copy-feedback";

export function PathInfoButton({
  path,
  title = "On-device location",
  description = "Where this model lives on disk.",
  className,
}: {
  path: string;
  title?: string;
  description?: string;
  className?: string;
}) {
  const [open, setOpen] = useState(false);
  const { copied, copy } = useCopyFeedback();

  const handleCopy = async (event: MouseEvent<HTMLButtonElement>) => {
    event.preventDefault();
    event.stopPropagation();
    await copy(path);
  };

  return (
    <>
      <Tooltip>
        <TooltipTrigger asChild={true}>
          <button
            type="button"
            aria-label="Show on-device path"
            onClick={(e) => {
              e.stopPropagation();
              setOpen(true);
            }}
            className={cn(
              "inline-flex size-7 shrink-0 cursor-pointer items-center justify-center rounded-full text-muted-foreground opacity-0 transition-[opacity,background-color,color] duration-150 hover:bg-muted hover:text-foreground focus-visible:opacity-100 group-hover/dl:opacity-100",
              className,
            )}
          >
            <HugeiconsIcon
              icon={FolderSearchIcon}
              strokeWidth={1.75}
              className="size-4"
            />
          </button>
        </TooltipTrigger>
        <TooltipContent side="bottom" className="tooltip-compact">
          Show path
        </TooltipContent>
      </Tooltip>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent
          className="sm:max-w-[520px]"
          onClick={(e) => e.stopPropagation()}
        >
          <DialogHeader>
            <DialogTitle>{title}</DialogTitle>
            <DialogDescription>{description}</DialogDescription>
          </DialogHeader>
          <div className="flex items-stretch gap-2">
            <div className="flex-1 select-text rounded-[10px] border border-border bg-muted/30 px-3 py-2.5 text-[12px] leading-5 text-foreground/85 break-all">
              {path}
            </div>
            <Tooltip>
              <TooltipTrigger asChild={true}>
                <button
                  type="button"
                  aria-label="Copy path"
                  onClick={handleCopy}
                  className="inline-flex shrink-0 cursor-pointer items-center justify-center rounded-[10px] border border-border bg-muted/30 px-3 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
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
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
