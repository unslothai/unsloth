// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { CheckIcon, FileTextIcon, LoaderIcon, XIcon } from "lucide-react";
import { Spinner } from "@/components/ui/spinner";
import { Badge } from "@/components/assistant-ui/badge";
import { cn } from "@/lib/utils";
import type { DocumentStatus } from "../types/rag";

const STATUS_LABEL: Record<DocumentStatus, string> = {
  pending: "Queued",
  running: "Indexing",
  completed: "Ready",
  failed: "Failed",
};

/** Compact status pill for a RAG document (spin / check / X), reused across lists. */
export function DocumentStatusChip({
  filename,
  status,
  progress,
  error,
  onRemove,
}: {
  filename: string;
  status: DocumentStatus;
  progress?: number | null;
  error?: string | null;
  onRemove?: () => void;
}) {
  const indexing = status === "pending" || status === "running";
  const pct =
    indexing && typeof progress === "number" && progress > 0
      ? ` ${Math.round(progress * 100)}%`
      : "";
  return (
    <Badge
      variant="outline"
      size="sm"
      title={error ?? filename}
      className={cn(
        "rounded-full inline-flex items-center gap-1.5 max-w-[16rem]",
        status === "failed" && "border-destructive/40 text-destructive",
      )}
    >
      {status === "completed" ? (
        <CheckIcon className="size-3 shrink-0" />
      ) : status === "failed" ? (
        <XIcon className="size-3 shrink-0" />
      ) : status === "running" ? (
        <Spinner className="size-3 shrink-0" />
      ) : (
        <LoaderIcon className="size-3 shrink-0 animate-spin" />
      )}
      <FileTextIcon className="size-3 shrink-0" />
      <span className="truncate">{filename}</span>
      <span className="shrink-0 text-[10px] text-muted-foreground">
        {STATUS_LABEL[status]}
        {pct}
      </span>
      {onRemove && (
        <button
          type="button"
          onClick={onRemove}
          aria-label={`Remove ${filename}`}
          className="shrink-0 rounded-full text-muted-foreground hover:text-foreground"
        >
          <XIcon className="size-3" />
        </button>
      )}
    </Badge>
  );
}
