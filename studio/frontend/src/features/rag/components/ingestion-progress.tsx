// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";
import { useIngestionEvents } from "../hooks/use-ingestion-events";

const STAGE_LABELS: Record<string, string> = {
  queued: "Queued",
  parse: "Parsing document",
  load_model: "Loading embedder",
  chunk: "Chunking text",
  embed: "Embedding chunks",
  done: "Indexing complete",
};

export function IngestionProgress({
  jobId,
  className,
}: {
  jobId: string;
  className?: string;
}) {
  const event = useIngestionEvents(jobId);
  if (!event) {
    return (
      <div className={cn("text-xs text-muted-foreground", className)}>
        Starting…
      </div>
    );
  }

  if (event.type === "error") {
    return (
      <div className={cn("text-xs text-destructive", className)}>
        {event.error}
      </div>
    );
  }

  if (event.type === "complete") {
    return (
      <div className={cn("text-xs text-muted-foreground", className)}>
        Indexed {event.num_chunks} chunks
      </div>
    );
  }

  const stage =
    "stage" in event && event.stage ? (event.stage as string) : "queued";
  const progress =
    "progress" in event && typeof event.progress === "number"
      ? event.progress
      : 0;
  const label = STAGE_LABELS[stage] ?? stage;

  return (
    <div className={cn("flex flex-col gap-1.5", className)}>
      <div className="flex items-center justify-between gap-3 text-xs text-muted-foreground">
        <span className="truncate">{label}</span>
        <span className="shrink-0 tabular-nums">{Math.round(progress * 100)}%</span>
      </div>
      <Progress value={Math.round(progress * 100)} className="h-1" />
    </div>
  );
}
