// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { Copy01Icon, Tick02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useState } from "react";
import { cn } from "@/lib/utils";

export function RevealKeyDialog({
  rawKey,
  onClose,
}: {
  rawKey: string | null;
  onClose: () => void;
}) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    if (!rawKey) return;
    if (copyToClipboard(rawKey)) {
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    }
  };

  return (
    <Dialog
      open={rawKey !== null}
      onOpenChange={(o) => {
        if (!o) {
          setCopied(false);
          onClose();
        }
      }}
    >
      <DialogContent
        className="max-w-lg"
        onEscapeKeyDown={(e) => e.preventDefault()}
        onPointerDownOutside={(e) => e.preventDefault()}
      >
        <DialogHeader>
          <DialogTitle>Your new API key</DialogTitle>
        </DialogHeader>
        <button
          type="button"
          onClick={handleCopy}
          className={cn(
            "flex w-full items-center justify-between gap-3 rounded-md border border-border bg-muted/40 p-4 font-mono text-sm transition-colors hover:bg-muted/60",
            copied && "border-emerald-500/40 bg-emerald-500/5",
          )}
        >
          <code className="min-w-0 flex-1 break-all text-left">{rawKey}</code>
          <HugeiconsIcon
            icon={copied ? Tick02Icon : Copy01Icon}
            className={cn("size-4 shrink-0", copied && "text-emerald-600")}
          />
        </button>
        <p className="text-xs text-muted-foreground">
          Store this key now — it won't be shown again after you close this dialog.
        </p>
        <DialogFooter>
          <Button onClick={onClose}>I've copied it</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
