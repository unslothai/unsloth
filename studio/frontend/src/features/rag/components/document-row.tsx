// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { Delete02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { KeyboardEvent, MouseEvent } from "react";
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
  onPreview,
  rightSlot,
  className,
}: {
  doc: RagDocument;
  onDelete?: () => void;
  /** Fired when the row body (not the delete button) is clicked.
   *  Per decision Q9, callers should only pass this for completed
   *  documents. */
  onPreview?: () => void;
  rightSlot?: React.ReactNode;
  className?: string;
}) {
  const isPreviewable = !!onPreview;

  const handleRowClick = (e: MouseEvent<HTMLDivElement>) => {
    if (!onPreview) return;
    // If a button/anchor/control was clicked (e.g. the delete icon),
    // skip preview — let that handler win. Buttons inside this row
    // additionally call stopPropagation, but this is defense in depth
    // for any descendant Button that forgets to.
    const target = e.target as HTMLElement | null;
    const interactive = target?.closest("button, a, [role=button]");
    if (interactive && interactive !== e.currentTarget) return;
    onPreview();
  };

  const handleRowKey = (e: KeyboardEvent<HTMLDivElement>) => {
    if (!onPreview) return;
    if (e.target !== e.currentTarget) return; // ignore child key events
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      onPreview();
    }
  };

  return (
    <div
      className={cn(
        "flex items-center justify-between gap-3 rounded-md border border-border/60 px-3 py-2",
        isPreviewable &&
          "cursor-pointer outline-none focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50 hover:bg-muted/40",
        className,
      )}
      {...(isPreviewable
        ? {
            role: "button",
            tabIndex: 0,
            onClick: handleRowClick,
            onKeyDown: handleRowKey,
            "aria-label": `Open preview of ${doc.filename}`,
          }
        : {})}
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
          onClick={(e) => {
            // Stop propagation so the row's onClick (preview open)
            // does not fire when the user is asking to delete.
            e.stopPropagation();
            onDelete();
          }}
          className="text-muted-foreground hover:text-destructive"
        >
          <HugeiconsIcon icon={Delete02Icon} size={16} />
        </Button>
      ) : null}
    </div>
  );
}
