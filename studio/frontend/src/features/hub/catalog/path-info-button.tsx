// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { usePlatformStore } from "@/config/env";
import { revealCachedModel } from "@/features/chat";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { Copy01Icon, Folder01Icon } from "@hugeicons/core-free-icons";
import { Tick02Icon } from "@/lib/tick-icon";
import { HugeiconsIcon } from "@hugeicons/react";
import type { MouseEvent } from "react";
import { useCopyFeedback } from "../hooks/use-copy-feedback";

/** Reveal a cached repo (or one GGUF variant's file) in the OS file manager.
 *  Resolved server-side from the HF cache, so only managed repos qualify. */
export function RevealPathButton({
  repoId,
  variant,
  className,
}: {
  repoId: string;
  variant?: string | null;
  className?: string;
}) {
  const deviceType = usePlatformStore((s) => s.deviceType);
  const revealLabel =
    deviceType === "mac"
      ? "Reveal in Finder"
      : deviceType === "windows"
        ? "Reveal in File Explorer"
        : "Reveal in File Manager";

  return (
    <Tooltip>
      <TooltipTrigger asChild={true}>
        <button
          type="button"
          aria-label={revealLabel}
          onClick={(e) => {
            e.stopPropagation();
            revealCachedModel(repoId, variant ?? undefined).catch((err) => {
              toast.error(
                err instanceof Error
                  ? err.message
                  : "Failed to open file manager",
              );
            });
          }}
          className={cn(
            "inline-flex size-7 shrink-0 cursor-pointer items-center justify-center rounded-full text-muted-foreground opacity-0 transition-[opacity,background-color,color] duration-150 hover:bg-muted hover:text-foreground focus-visible:opacity-100 group-hover/dl:opacity-100",
            className,
          )}
        >
          <HugeiconsIcon
            icon={Folder01Icon}
            strokeWidth={1.75}
            className="size-4"
          />
        </button>
      </TooltipTrigger>
      <TooltipContent side="bottom" className="tooltip-compact">
        {revealLabel}
      </TooltipContent>
    </Tooltip>
  );
}

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
