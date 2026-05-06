// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { cn } from "@/lib/utils";
import { Copy01Icon, Tick02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useState } from "react";

export function KeyRevealCard({
  rawKey,
  onDone,
}: {
  rawKey: string;
  onDone: () => void;
}) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    if (await copyToClipboard(rawKey)) {
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    }
  };

  return (
    <div className="flex flex-col gap-2 rounded-lg border border-emerald-500/30 bg-emerald-500/5 p-3">
      <div className="flex items-center gap-1.5">
        <HugeiconsIcon
          icon={Tick02Icon}
          className="size-3.5 text-emerald-600 dark:text-emerald-500"
        />
        <span className="text-xs font-medium text-emerald-700 dark:text-emerald-500">
          New access token created
        </span>
      </div>
      <button
        type="button"
        onClick={handleCopy}
        className={cn(
          "flex w-full items-center justify-between gap-3 rounded-md border border-border bg-muted/40 px-3 py-2.5 font-mono text-sm transition-colors hover:bg-muted/60",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 focus-visible:ring-offset-background",
          copied && "border-emerald-500/40 bg-emerald-500/10",
        )}
        aria-label={copied ? "Access token copied" : "Copy access token"}
      >
        <code className="min-w-0 flex-1 break-all text-left text-foreground">
          {rawKey}
        </code>
        <HugeiconsIcon
          icon={copied ? Tick02Icon : Copy01Icon}
          className={cn("size-4 shrink-0", copied && "text-emerald-600")}
        />
      </button>
      <div className="flex items-center justify-between gap-3 pt-0.5">
        <p className="text-[11px] text-muted-foreground">
          Copy now — this won't be shown again.
        </p>
        <Button
          type="button"
          size="sm"
          onClick={onDone}
          className="focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 focus-visible:ring-offset-background"
        >
          Done
        </Button>
      </div>
    </div>
  );
}
