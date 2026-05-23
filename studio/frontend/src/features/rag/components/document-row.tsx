// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { Delete02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { RagDocument } from "../api/rag-api";

const STATUS_VARIANT: Record<
  RagDocument["status"],
  "default" | "secondary" | "destructive" | "outline"
> = {
  pending: "outline",
  running: "secondary",
  completed: "default",
  failed: "destructive",
};

function humanBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function DocumentRow({
  doc,
  onDelete,
  rightSlot,
  className,
}: {
  doc: RagDocument;
  onDelete?: () => void;
  rightSlot?: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex items-center justify-between gap-3 rounded-md border border-border/60 px-3 py-2",
        className,
      )}
    >
      <div className="flex min-w-0 flex-1 flex-col gap-1">
        <div className="flex items-center gap-2">
          <span className="truncate text-sm font-medium" title={doc.filename}>
            {doc.filename}
          </span>
          <Badge variant={STATUS_VARIANT[doc.status]} className="capitalize">
            {doc.status}
          </Badge>
        </div>
        <div className="flex gap-3 text-xs text-muted-foreground">
          <span>{humanBytes(doc.byte_size)}</span>
          {doc.status === "completed" ? (
            <span>{doc.num_chunks} chunks</span>
          ) : null}
          {doc.error ? (
            <span className="text-destructive">{doc.error}</span>
          ) : null}
        </div>
        {rightSlot}
      </div>
      {onDelete ? (
        <Button
          variant="ghost"
          size="icon"
          aria-label="Delete document"
          onClick={onDelete}
          className="text-muted-foreground hover:text-destructive"
        >
          <HugeiconsIcon icon={Delete02Icon} size={16} />
        </Button>
      ) : null}
    </div>
  );
}
