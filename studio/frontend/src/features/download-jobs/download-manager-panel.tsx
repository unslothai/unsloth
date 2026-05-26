// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import {
  Alert02Icon,
  ArrowDown01Icon,
  ArrowUp01Icon,
  Cancel01Icon,
  CheckmarkCircle02Icon,
  Download01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useMemo, useState } from "react";
import { DownloadProgressBar } from "./download-progress-bar";
import {
  type ManagedDownload,
  downloadManager,
  hydrateDownloadManager,
  useDownloadManagerStore,
} from "./download-manager-store";

function createOrderedJobKeysSelector(): (state: {
  jobs: Record<string, ManagedDownload>;
}) => string[] {
  let cache: { signature: string; keys: string[] } = {
    signature: "",
    keys: [],
  };
  return (state) => {
    const ordered = Object.values(state.jobs)
      .map((job) => ({ key: job.key, startedAt: job.startedAt }))
      .sort((a, b) => a.startedAt - b.startedAt);
    const signature = ordered
      .map((job) => `${job.key}\u0001${job.startedAt}`)
      .join("\u0002");
    if (signature === cache.signature) {
      return cache.keys;
    }
    const keys = ordered.map((job) => job.key);
    cache = { signature, keys };
    return keys;
  };
}

function selectActiveJobCount(state: {
  jobs: Record<string, ManagedDownload>;
}): number {
  let count = 0;
  for (const job of Object.values(state.jobs)) {
    if (job.state === "running" || job.state === "cancelling") count += 1;
  }
  return count;
}

function variantSuffix(job: ManagedDownload): string {
  return job.variant ? ` · ${job.variant}` : "";
}

function StatusLine({ job }: { job: ManagedDownload }) {
  if (job.state === "complete") {
    return <span className="text-status-success">Downloaded</span>;
  }
  if (job.state === "cancelled") {
    return <span>Cancelled. Partial files kept.</span>;
  }
  if (job.state === "error") {
    return (
      <span className="text-destructive">{job.error ?? "Download failed"}</span>
    );
  }
  if (job.state === "cancelling") {
    return <span>Cancelling…</span>;
  }
  return null;
}

function DownloadRow({ jobKey }: { jobKey: string }) {
  const job = useDownloadManagerStore((state) => state.jobs[jobKey]);
  if (!job) return null;
  const active = job.state === "running" || job.state === "cancelling";
  const terminal =
    job.state === "complete" ||
    job.state === "cancelled" ||
    job.state === "error";
  return (
    <li className="flex flex-col gap-1.5 px-3 py-2.5">
      <div className="flex items-center gap-2">
        <span className="min-w-0 flex-1 truncate text-[12.5px] font-medium text-foreground">
          {job.repoId}
          <span className="text-muted-foreground">{variantSuffix(job)}</span>
        </span>
        {job.state === "complete" && (
          <HugeiconsIcon
            icon={CheckmarkCircle02Icon}
            strokeWidth={2}
            className="size-4 shrink-0 text-status-success"
          />
        )}
        {job.state === "error" && (
          <HugeiconsIcon
            icon={Alert02Icon}
            strokeWidth={2}
            className="size-4 shrink-0 text-destructive"
          />
        )}
        <Tooltip>
          <TooltipTrigger asChild={true}>
            <button
              type="button"
              aria-label={active ? "Cancel download" : "Dismiss"}
              disabled={job.state === "cancelling"}
              onClick={() =>
                active
                  ? void downloadManager.cancel(job.key)
                  : downloadManager.dismiss(job.key)
              }
              className={cn(
                "inline-flex size-6 shrink-0 cursor-pointer items-center justify-center rounded-[7px] text-muted-foreground transition-colors",
                "hover:bg-foreground/[0.06] hover:text-foreground disabled:cursor-default disabled:opacity-50 dark:hover:bg-white/[0.06]",
              )}
            >
              <HugeiconsIcon
                icon={Cancel01Icon}
                strokeWidth={1.75}
                className="size-3.5"
              />
            </button>
          </TooltipTrigger>
          <TooltipContent side="top" sideOffset={4}>
            {active ? "Cancel download" : "Dismiss"}
          </TooltipContent>
        </Tooltip>
      </div>
      {active ? (
        <DownloadProgressBar
          progress={{
            expectedBytes: job.expectedBytes,
            downloadedBytes: job.downloadedBytes,
            fraction: job.fraction,
          }}
          bytesPerSec={job.bytesPerSec}
        />
      ) : null}
      {terminal || job.state === "cancelling" ? (
        <div className="px-0 text-[11px] text-muted-foreground tabular-nums">
          <StatusLine job={job} />
        </div>
      ) : null}
    </li>
  );
}

export function DownloadManagerPanel() {
  const [collapsed, setCollapsed] = useState(false);

  useEffect(() => {
    hydrateDownloadManager();
  }, []);

  const selectOrderedJobKeys = useMemo(createOrderedJobKeysSelector, []);
  const jobKeys = useDownloadManagerStore(selectOrderedJobKeys);
  const activeCount = useDownloadManagerStore(selectActiveJobCount);

  if (jobKeys.length === 0) return null;

  const headerLabel =
    activeCount > 0
      ? `Downloading ${activeCount} ${activeCount === 1 ? "item" : "items"}`
      : "Downloads";

  return (
    <div className="pointer-events-none fixed bottom-4 right-4 z-50 w-[min(360px,calc(100vw-2rem))]">
      <div className="pointer-events-auto overflow-hidden rounded-[14px] border border-border bg-popover shadow-lg">
        <div className="flex items-center gap-2 border-b border-border px-3 py-2">
          <HugeiconsIcon
            icon={Download01Icon}
            strokeWidth={1.75}
            className="size-4 shrink-0 text-muted-foreground"
          />
          <span className="min-w-0 flex-1 truncate text-[12.5px] font-semibold text-foreground">
            {headerLabel}
          </span>
          <button
            type="button"
            aria-label={collapsed ? "Expand downloads" : "Collapse downloads"}
            onClick={() => setCollapsed((c) => !c)}
            className="inline-flex size-6 shrink-0 cursor-pointer items-center justify-center rounded-[7px] text-muted-foreground transition-colors hover:bg-foreground/[0.06] hover:text-foreground dark:hover:bg-white/[0.06]"
          >
            <HugeiconsIcon
              icon={collapsed ? ArrowUp01Icon : ArrowDown01Icon}
              strokeWidth={1.75}
              className="size-3.5"
            />
          </button>
        </div>
        {!collapsed && (
          <ul className="max-h-[60vh] divide-y divide-border overflow-y-auto [scrollbar-width:thin]">
            {jobKeys.map((jobKey) => (
              <DownloadRow key={jobKey} jobKey={jobKey} />
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
