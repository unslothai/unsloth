// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/assistant-ui/badge";
import { Spinner } from "@/components/ui/spinner";
import { cn } from "@/lib/utils";
import { FileEmpty02Icon, Folder02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { XIcon } from "lucide-react";
import type { DocumentStatus } from "../types/rag";

export const STAGE_LABELS: Record<string, string> = {
  parsing: "Reading document",
  ocr: "Reading scanned pages",
  captioning: "Reading charts and figures",
  chunking: "Preparing text",
  embedding: "Indexing text",
  storing: "Saving document",
};

export function DocumentStatusChip({
  filename,
  status,
  progress,
  stage,
  error,
  onRemove,
  onRetry,
  shared = false,
}: {
  filename: string;
  status: DocumentStatus;
  progress?: number | null;
  stage?: string | null;
  error?: string | null;
  onRemove?: () => void;
  /** Offered on a failed row whose original file is still in hand. */
  onRetry?: () => void;
  /** Indexed for the whole project rather than this one chat: swap the file
   * glyph for a folder so the two scopes are told apart at a glance. */
  shared?: boolean;
}) {
  const processing = status === "pending" || status === "running";
  return (
    <Badge
      variant="outline"
      size="sm"
      title={
        error ??
        (processing && stage && STAGE_LABELS[stage]
          ? `${filename} — ${STAGE_LABELS[stage]}`
          : shared
            ? `${filename} — shared with every chat in this project`
            : filename)
      }
      className={cn(
        "rounded-full inline-flex items-center gap-1.5 max-w-[calc(16rem*var(--ui-space-scale,1))]",
        status === "failed" && "border-destructive/40 text-destructive",
      )}
    >
      {/* file, or folder when the doc is a project-wide source */}
      <HugeiconsIcon
        icon={shared ? Folder02Icon : FileEmpty02Icon}
        strokeWidth={2}
        className="size-3 shrink-0"
      />
      <span className="truncate">{filename}</span>
      {status === "failed" && onRetry ? (
        <button
          type="button"
          onClick={onRetry}
          aria-label={`Retry ${filename}`}
          className="shrink-0 text-ui-10 font-medium underline-offset-2 hover:underline"
        >
          Retry
        </button>
      ) : null}
      {/* spinner while indexing, else close button */}
      {processing ? (
        <span className="flex shrink-0 items-center gap-1 text-ui-10 text-muted-foreground">
          {progress != null
            ? `${Math.round(progress <= 1 ? progress * 100 : progress)}%`
            : null}
          <Spinner className="size-3.5" />
        </span>
      ) : onRemove ? (
        <button
          type="button"
          onClick={onRemove}
          aria-label={`Remove ${filename}`}
          className="shrink-0 rounded-full text-muted-foreground hover:text-foreground"
        >
          <XIcon className="size-3" />
        </button>
      ) : null}
    </Badge>
  );
}
