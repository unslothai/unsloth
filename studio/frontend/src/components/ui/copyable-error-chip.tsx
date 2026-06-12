// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { cn } from "@/lib/utils";
import { Copy01Icon, Tick02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useRef, useState } from "react";

interface CopyableErrorChipProps {
  message: string;
  className?: string;
}

export function CopyableErrorChip({
  message,
  className,
}: CopyableErrorChipProps) {
  const [copied, setCopied] = useState(false);
  const resetTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Clear any pending reset on unmount to avoid a setState on an
  // unmounted component.
  useEffect(() => () => {
    if (resetTimer.current) clearTimeout(resetTimer.current);
  }, []);

  const handleCopy = async () => {
    if (await copyToClipboard(message)) {
      setCopied(true);
      if (resetTimer.current) clearTimeout(resetTimer.current);
      resetTimer.current = setTimeout(() => {
        setCopied(false);
        resetTimer.current = null;
      }, 1800);
    }
  };

  return (
    <Popover>
      <PopoverTrigger asChild={true}>
        {/* No aria-label override: the visible message text is the
            button's accessible name, so screen readers announce the
            full (untruncated) error. Truncation here is purely visual. */}
        <button
          type="button"
          className={cn(
            "flex max-w-[28rem] min-w-0 cursor-pointer items-center rounded-md text-left text-xs text-destructive transition-colors hover:bg-destructive/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
            className,
          )}
        >
          <span className="min-w-0 flex-1 truncate">{message}</span>
        </button>
      </PopoverTrigger>
      <PopoverContent
        align="start"
        side="bottom"
        className="w-[min(36rem,calc(100vw-1rem))] gap-2"
      >
        <div className="flex items-start justify-between gap-2">
          <span className="text-xs font-medium text-destructive">Error</span>
          <button
            type="button"
            onClick={handleCopy}
            aria-label={copied ? "Copied" : "Copy error message"}
            className={cn(
              "inline-flex items-center gap-1 rounded-md border border-border/60 px-2 py-1 text-[11px] text-muted-foreground transition-colors hover:bg-muted/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
              copied && "border-emerald-500/40 text-emerald-600 dark:text-emerald-500",
            )}
          >
            <HugeiconsIcon
              icon={copied ? Tick02Icon : Copy01Icon}
              className="size-3.5"
            />
            {copied ? "Copied" : "Copy"}
          </button>
        </div>
        <p className="max-h-64 overflow-y-auto select-text whitespace-pre-wrap break-words text-xs text-destructive">
          {message}
        </p>
      </PopoverContent>
    </Popover>
  );
}
