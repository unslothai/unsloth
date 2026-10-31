// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { LineBox } from "@/components/ui/line-box";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { cn } from "@/lib/utils";
import { Tick02Icon } from "@/lib/tick-icon";
import { Copy01Icon } from "@/lib/copy-icon";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useRef, useState } from "react";

export function UrlPill({ url, className }: { url: string; className?: string }) {
  const [copied, setCopied] = useState(false);
  const copyTimer = useRef<number | null>(null);
  useEffect(() => {
    return () => {
      if (copyTimer.current !== null) {
        window.clearTimeout(copyTimer.current);
      }
    };
  }, []);
  return (
    <div className={cn("relative min-w-0", className)}>
      <LineBox
        as="code"
        className="block w-full break-all border border-border bg-muted/40 py-2 pr-10 pl-3 font-mono text-xs text-foreground"
      >
        {url}
      </LineBox>
      <button
        type="button"
        aria-label={copied ? "Copied" : `Copy ${url}`}
        title={copied ? "Copied" : "Copy URL"}
        onClick={async () => {
          if (!(await copyToClipboard(url))) {
            return;
          }
          setCopied(true);
          if (copyTimer.current !== null) {
            window.clearTimeout(copyTimer.current);
          }
          copyTimer.current = window.setTimeout(() => setCopied(false), 1800);
        }}
        className="absolute top-1/2 right-1.5 flex size-6 -translate-y-1/2 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-accent hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
      >
        <HugeiconsIcon
          icon={copied ? Tick02Icon : Copy01Icon}
          className={cn("size-3.5", copied && "text-control-accent")}
        />
      </button>
    </div>
  );
}
