// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { hasAuthToken, mustChangePassword } from "@/features/auth/session";
import { isTauri } from "@/lib/api-base";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { cn } from "@/lib/utils";
import {
  Alert02Icon,
  Cancel01Icon,
  CheckmarkCircle02Icon,
  Download01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useRouterState } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
import {
  type ManagedDownload,
  downloadManager,
  hydrateDownloadManager,
  useDownloadManagerStore,
} from "./download-manager-controller";
import { DownloadProgressBar } from "./download-progress-bar";

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

function canUseDownloadManager(pathname: string): boolean {
  if (isTauri) return true;
  if (
    pathname === "/login" ||
    pathname === "/change-password" ||
    pathname === "/signup"
  ) {
    return false;
  }
  return hasAuthToken() && !mustChangePassword();
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
  if (job.error) {
    return <span className="text-status-warning">{job.error}</span>;
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
    <li className="flex flex-col gap-1.5 py-2.5 pl-4 pr-3">
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
                "inline-flex size-6 shrink-0 cursor-pointer items-center justify-center rounded-full text-muted-foreground transition-colors",
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
      {terminal || job.state === "cancelling" || job.error ? (
        <div className="px-0 text-[11px] text-muted-foreground tabular-nums">
          <StatusLine job={job} />
        </div>
      ) : null}
    </li>
  );
}

export function DownloadManagerPanel({
  positioned = true,
}: { positioned?: boolean } = {}) {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const enabled = canUseDownloadManager(pathname);
  const [collapsed, setCollapsed] = useState(false);

  useEffect(() => {
    if (!enabled) return;
    hydrateDownloadManager();
  }, [enabled]);

  const selectOrderedJobKeys = useMemo(createOrderedJobKeysSelector, []);
  const jobKeys = useDownloadManagerStore(selectOrderedJobKeys);
  const activeCount = useDownloadManagerStore(selectActiveJobCount);

  if (!enabled || jobKeys.length === 0) return null;

  const headerLabel =
    activeCount > 0
      ? `Downloading ${activeCount} ${activeCount === 1 ? "item" : "items"}`
      : "Downloads";

  return (
    <div
      className={cn(
        // Standalone: anchor bottom-right. In a shared stack (positioned=false)
        // flow as a right-aligned row so overlays stack instead of overlapping.
        "pointer-events-none",
        positioned ? "fixed bottom-4 right-4 z-50" : "flex justify-end",
      )}
    >
      {collapsed ? (
        <Tooltip>
          <TooltipTrigger asChild={true}>
            <button
              type="button"
              aria-label={headerLabel}
              onClick={() => setCollapsed(false)}
              className="hub-download-fab pointer-events-auto"
            >
              <HugeiconsIcon
                icon={Download01Icon}
                strokeWidth={1.75}
                className="size-[18px]"
              />
              {activeCount > 0 && (
                <span className="hub-download-fab-badge">{activeCount}</span>
              )}
            </button>
          </TooltipTrigger>
          <TooltipContent side="left" sideOffset={6}>
            {headerLabel}
          </TooltipContent>
        </Tooltip>
      ) : (
        <div className="hub-download-panel pointer-events-auto w-[min(400px,calc(100vw-2rem))] overflow-hidden">
          <div className="flex items-center gap-2 border-b border-foreground/[0.07] py-2 pl-4 pr-3">
            <span className="min-w-0 flex-1 truncate text-[12.5px] font-semibold text-foreground">
              {headerLabel}
            </span>
            <button
              type="button"
              aria-label="Collapse downloads"
              onClick={() => setCollapsed(true)}
              className="inline-flex size-6 shrink-0 cursor-pointer items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-foreground/[0.06] hover:text-foreground dark:hover:bg-white/[0.06]"
            >
              <HugeiconsIcon
                icon={ChevronDownStandardIcon}
                strokeWidth={1.75}
                className="size-3.5"
              />
            </button>
          </div>
          <ul className="max-h-[60vh] divide-y divide-foreground/[0.06] overflow-y-auto [scrollbar-width:thin]">
            {jobKeys.map((jobKey) => (
              <DownloadRow key={jobKey} jobKey={jobKey} />
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
