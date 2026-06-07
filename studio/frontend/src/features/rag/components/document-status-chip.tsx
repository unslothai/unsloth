// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { XIcon } from "lucide-react";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  CheckmarkCircle02Icon,
  File02Icon,
  Loading03Icon,
} from "@hugeicons/core-free-icons";
import { Badge } from "@/components/assistant-ui/badge";
import { cn } from "@/lib/utils";
import type { DocumentStatus } from "../types/rag";

/** Status pill for a RAG document, reused across lists. State is shown by the
 * leading icon alone (a spinner matching the toasts while processing, a check
 * when ready); no status word. */
export function DocumentStatusChip({
  filename,
  status,
  error,
  onRemove,
}: {
  filename: string;
  status: DocumentStatus;
  progress?: number | null;
  error?: string | null;
  onRemove?: () => void;
}) {
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
        <HugeiconsIcon
          icon={CheckmarkCircle02Icon}
          strokeWidth={2}
          className="size-3 shrink-0"
        />
      ) : status === "failed" ? (
        <XIcon className="size-3 shrink-0" />
      ) : (
        // Same spinner as the toast notifications (sonner loading icon).
        <HugeiconsIcon
          icon={Loading03Icon}
          strokeWidth={2}
          className="size-3 shrink-0 animate-spin"
        />
      )}
      <HugeiconsIcon
        icon={File02Icon}
        strokeWidth={2}
        className="size-3 shrink-0"
      />
      <span className="truncate">{filename}</span>
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
