// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useHubDownloadQueue, useQueuedHubEntries } from "./use-hub-download-queue";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useEngines } from "@/features/model-picker/hooks/use-engines";
import {
  audioCppDisplayName,
  isAudioCppFolderId,
} from "../../audio/audio-cpp-catalog";
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
import { presentedProgress } from "./download-presentation";
import { requiredAssetKind } from "./required-assets";

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

/** The repo as a row names it. A package folder of the shared GGUF audio repo is known by its
 *  folder name, as the Hub and the pickers show it; every other id reads as itself. */
function repoLabel(repoId: string): string {
  return isAudioCppFolderId(repoId) ? audioCppDisplayName(repoId) : repoId;
}

/** True for a companion repo a staged media pick needs (text encoder, VAE, configs), false for the
 *  model file itself and for every plain download. */
function isRequiredAssetJob(job: ManagedDownload): boolean {
  if (!job.variant?.startsWith("@")) {
    return false;
  }
  // The staging page tagged the entry it picked, which is the only reliable answer: a checkpoint
  // can be a curated single .safetensors and companion repos carry .safetensors too, so the
  // extension decides nothing. The old guess stays for jobs persisted before the flag existed,
  // which would otherwise change label mid-download after a restart.
  const isModelFile =
    job.checkpoint ??
    job.scopedFiles?.some((file) => file.toLowerCase().endsWith(".gguf"));
  return !isModelFile;
}

/** Name a companion by what its files are, so a second download after Run does not read as the
 *  model being fetched again. */
function requiredAssetSuffix(files: readonly string[] | undefined): string {
  return ` · ${requiredAssetKind(files) ?? "Required assets"}`;
}

const REQUIRED_ASSET_NOTE =
  "Required to run this model. Downloaded once and shared by every quant.";

function variantSuffix(job: ManagedDownload): string {
  if (job.variant?.startsWith("@")) {
    return isRequiredAssetJob(job)
      ? requiredAssetSuffix(job.scopedFiles)
      : " · Model file";
  }
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

/** The drafter file a presented companion transfers, or what a media pick's companion is for. */
function RowDetail({ job }: { job: ManagedDownload }) {
  if (job.presentation) {
    return (
      <div className="truncate text-ui-10p5 text-muted-foreground">
        {job.presentation.filename}
      </div>
    );
  }
  if (isRequiredAssetJob(job)) {
    return (
      <div className="text-ui-10p5 text-muted-foreground">
        {REQUIRED_ASSET_NOTE}
      </div>
    );
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
  const progress = presentedProgress(job);
  return (
    <li className="flex flex-col gap-1.5 py-2.5 pl-4 pr-3">
      <div className="flex items-center gap-2">
        <span className="min-w-0 flex-1 truncate text-ui-12p5 font-medium text-foreground">
          {job.presentation?.label ?? repoLabel(job.repoId)}
          <span className="text-muted-foreground">
            {job.presentation
              ? ` · ${repoLabel(job.repoId)}`
              : variantSuffix(job)}
          </span>
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
                "hover:bg-[color-mix(in_oklab,var(--foreground)_calc(6%*var(--contrast-wash-gain,1)),transparent)] hover:text-foreground disabled:cursor-default disabled:opacity-50 dark:hover:bg-[rgb(255_255_255_/_calc(0.06*var(--contrast-wash-gain,1)))]",
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
      <RowDetail job={job} />
      {active ? (
        <DownloadProgressBar
          progress={progress}
          bytesPerSec={job.bytesPerSec}
          cancelling={job.state === "cancelling"}
          etaSeconds={job.etaSeconds}
          activity={job.activity}
        />
      ) : null}
      {job.details?.length ? (
        <details className="text-ui-11">
          <summary>Installation details</summary>
          <pre className="max-h-40 overflow-auto whitespace-pre-wrap break-all">
            {job.details.join("\n")}
          </pre>
        </details>
      ) : null}
      {terminal || job.state === "cancelling" || job.error ? (
        <div className="px-0 text-ui-11 text-muted-foreground tabular-nums">
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
  useEngines(enabled, true);
  useHubDownloadQueue();
  const [collapsed, setCollapsed] = useState(false);

  useEffect(() => {
    if (!enabled) return;
    hydrateDownloadManager();
  }, [enabled]);

  const selectOrderedJobKeys = useMemo(createOrderedJobKeysSelector, []);
  const jobKeys = useDownloadManagerStore(selectOrderedJobKeys);
  const queued = useQueuedHubEntries();
  const activeCount = useDownloadManagerStore(selectActiveJobCount) + queued.length;

  if (!enabled || (jobKeys.length === 0 && queued.length === 0)) return null;

  const headerLabel =
    activeCount > 0
      ? `Downloading ${activeCount} ${activeCount === 1 ? "item" : "items"}`
      : "Downloads";

  return (
    <div
      className={cn(
        // Standalone: anchor bottom-right. In a shared stack (positioned=false)
        // flow as a right-aligned row so overlays stack instead of overlapping.
        // min-h-0 there: a flex item's min-height defaults to auto, so the capped
        // stack would squeeze the update card instead of this list.
        "pointer-events-none",
        positioned ? "fixed bottom-4 right-4 z-50" : "flex min-h-0 justify-end",
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
                className="size-[calc(18px*var(--ui-space-scale,1))]"
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
        <div className="hub-download-panel pointer-events-auto flex min-h-0 w-[min(400px,calc(100vw-2rem))] flex-col overflow-hidden">
          <div className="flex items-center gap-2 border-b border-[color-mix(in_oklab,var(--foreground)_calc(7%*var(--contrast-edge-gain,1)),transparent)] py-2 pl-4 pr-3">
            <span className="min-w-0 flex-1 truncate text-ui-12p5 font-semibold text-foreground">
              {headerLabel}
            </span>
            <button
              type="button"
              aria-label="Collapse downloads"
              onClick={() => setCollapsed(true)}
              className="inline-flex size-6 shrink-0 cursor-pointer items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-[color-mix(in_oklab,var(--foreground)_calc(6%*var(--contrast-wash-gain,1)),transparent)] hover:text-foreground dark:hover:bg-[rgb(255_255_255_/_calc(0.06*var(--contrast-wash-gain,1)))]"
            >
              <HugeiconsIcon
                icon={ChevronDownStandardIcon}
                strokeWidth={1.75}
                className="size-3.5"
              />
            </button>
          </div>
          <ul className="max-h-[60dvh] divide-y divide-foreground/[0.06] overflow-y-auto [scrollbar-width:thin]">
            {jobKeys.map((jobKey) => (
              <DownloadRow key={jobKey} jobKey={jobKey} />
            ))}
            {queued.map((entry, i) => <li key={`${entry.planId}:${i}`} className="flex flex-col gap-1.5 py-2.5 pl-4 pr-3">
              <span className="truncate text-ui-12p5 font-medium">{entry.repoId}<span className="text-muted-foreground">{entry.checkpoint !== false ? " · Model file" : requiredAssetSuffix(entry.files)}</span></span>
              <span className="text-ui-11 text-muted-foreground">{entry.checkpoint !== false ? "Queued" : `Queued · ${REQUIRED_ASSET_NOTE}`}</span>
            </li>)}
          </ul>
        </div>
      )}
    </div>
  );
}
