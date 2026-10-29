// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ArrowDown, ArrowUp, X } from "lucide-react";
import type { ReactNode } from "react";

import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";

import type { ImageQueueJob } from "./api";
import { promptSnippet, queueJobSummary, queuedJobs, runningJob } from "./lib/generation-queue";

/** The running job and the ones lined up behind it. Finished jobs leave the list; their images are in the gallery. */
export function GenerationQueuePanel({
  jobs,
  onCancel,
  onMove,
  onClear,
}: {
  jobs: ImageQueueJob[];
  onCancel: (job: ImageQueueJob) => void;
  onMove: (job: ImageQueueJob, index: number) => void;
  onClear: () => void;
}) {
  const running = runningJob(jobs);
  const queued = queuedJobs(jobs);
  if (!running && queued.length === 0) return null;
  return (
    <div className="mx-4 mb-2 rounded-lg border bg-muted/30 text-xs" data-testid="images-queue-panel">
      <div className="flex items-center justify-between px-3 pt-2 pb-1">
        <span className="font-medium">Queue · {queued.length} waiting</span>
        {queued.length > 0 && (
          <Button variant="ghost" size="sm" className="h-6 px-2 text-xs" onClick={onClear}>
            Clear
          </Button>
        )}
      </div>
      <ol className="max-h-48 overflow-y-auto pb-1">
        {running && (
          <QueueRow job={running} label={<Spinner className="size-3" />}>
            <RowButton label="Stop this job" onClick={() => onCancel(running)}>
              <X className="size-3.5" />
            </RowButton>
          </QueueRow>
        )}
        {queued.map((job, index) => (
          <QueueRow key={job.id} job={job} label={index + 1}>
            <RowButton label="Move up" disabled={index === 0} onClick={() => onMove(job, index - 1)}>
              <ArrowUp className="size-3.5" />
            </RowButton>
            <RowButton
              label="Move down"
              disabled={index === queued.length - 1}
              onClick={() => onMove(job, index + 1)}
            >
              <ArrowDown className="size-3.5" />
            </RowButton>
            <RowButton label="Remove from queue" onClick={() => onCancel(job)}>
              <X className="size-3.5" />
            </RowButton>
          </QueueRow>
        ))}
      </ol>
    </div>
  );
}

function QueueRow({
  job,
  label,
  children,
}: {
  job: ImageQueueJob;
  label: ReactNode;
  children: ReactNode;
}) {
  return (
    <li className="flex items-center gap-2 px-3 py-1" data-status={job.status}>
      <span className="flex w-4 shrink-0 justify-center text-muted-foreground tabular-nums">{label}</span>
      <div className="min-w-0 flex-1">
        <div className="truncate" title={job.prompt}>
          {promptSnippet(job.prompt)}
        </div>
        <div className="text-muted-foreground">{queueJobSummary(job)}</div>
      </div>
      <div className="flex shrink-0 items-center">{children}</div>
    </li>
  );
}

function RowButton({
  label,
  disabled,
  onClick,
  children,
}: {
  label: string;
  disabled?: boolean;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <Button
      variant="ghost"
      size="icon"
      className="size-6"
      aria-label={label}
      title={label}
      disabled={disabled}
      onClick={onClick}
    >
      {children}
    </Button>
  );
}
