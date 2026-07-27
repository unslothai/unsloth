// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { XIcon } from "lucide-react";
import { Spinner } from "@/components/ui/spinner";
import { HugeiconsIcon } from "@hugeicons/react";
import { File02Icon } from "@hugeicons/core-free-icons";
import { Badge } from "@/components/assistant-ui/badge";
import { cn } from "@/lib/utils";
import type { DocumentStatus } from "../types/rag";

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
  const processing = status === "pending" || status === "running";
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
      {/* file */}
      <HugeiconsIcon
        icon={File02Icon}
        strokeWidth={2}
        className="size-3 shrink-0"
      />
      <span className="truncate">{filename}</span>
      {/* spinner while indexing, else close button */}
      {processing ? (
        <Spinner className="shrink-0 size-3.5 text-muted-foreground" />
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
