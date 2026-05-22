// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  CloudOffIcon,
  CubeIcon,
  FilterIcon,
  RefreshIcon,
  WifiDisconnected02Icon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { HugeiconsIcon } from "@hugeicons/react";

export function NetworkErrorState({
  online,
  message,
  onRetry,
}: {
  online: boolean;
  message: string;
  onRetry: () => void;
}) {
  const title = online ? "Couldn't reach Hugging Face" : "You're offline";
  const body = online
    ? "The discovery feed couldn't load. Check your connection or try again."
    : "Reconnect to the internet to browse models from Hugging Face.";
  const icon = online ? CloudOffIcon : WifiDisconnected02Icon;

  return (
    <div className="flex min-h-[260px] flex-col items-center justify-center gap-3 px-6 text-center">
      <div className="inline-flex size-11 items-center justify-center rounded-[12px] bg-amber-500/10 text-amber-700 dark:text-amber-300">
        <HugeiconsIcon icon={icon} strokeWidth={1.6} className="size-5" />
      </div>
      <div className="space-y-1">
        <p className="text-[14px] font-semibold tracking-tight text-foreground">
          {title}
        </p>
        <p className="max-w-md text-[12.5px] leading-5 text-muted-foreground">
          {body}
        </p>
        <p className="text-[11px] text-muted-foreground/70">{message}</p>
      </div>
      <button
        type="button"
        onClick={onRetry}
        className="inline-flex h-8 items-center gap-1.5 rounded-[10px] bg-transparent px-3 text-[12px] font-medium text-foreground transition-colors hover:bg-foreground/[0.04] dark:hover:bg-white/[0.05]"
      >
        <HugeiconsIcon
          icon={RefreshIcon}
          strokeWidth={1.75}
          className="size-3.5"
        />
        Try again
      </button>
    </div>
  );
}

export function FilterStarvedState({
  scannedCount,
  hasActiveFilters,
  onKeepSearching,
  onClearFilters,
}: {
  scannedCount: number;
  hasActiveFilters: boolean;
  onKeepSearching: () => void;
  onClearFilters: () => void;
}) {
  return (
    <div className="flex min-h-[260px] flex-col items-center justify-center gap-3 px-6 text-center">
      <div className="inline-flex size-11 items-center justify-center rounded-[12px] bg-muted text-muted-foreground">
        <HugeiconsIcon icon={FilterIcon} strokeWidth={1.5} className="size-5" />
      </div>
      <div className="space-y-1">
        <p className="text-[14px] font-semibold tracking-tight text-foreground">
          No matches yet
        </p>
        <p className="max-w-md text-[12.5px] leading-5 text-muted-foreground">
          Scanned {scannedCount.toLocaleString()} results. None match the
          current filters. Loosen them, or keep searching for a deeper sweep.
        </p>
      </div>
      <div className="flex flex-wrap items-center justify-center gap-2 pt-1">
        {hasActiveFilters && (
          <button
            type="button"
            onClick={onClearFilters}
            className="inline-flex h-8 items-center gap-1.5 rounded-[10px] bg-foreground/[0.06] px-3 text-[12px] font-medium text-foreground transition-colors hover:bg-foreground/[0.1] dark:bg-white/[0.06] dark:hover:bg-white/[0.1]"
          >
            Clear filters
          </button>
        )}
        <button
          type="button"
          onClick={onKeepSearching}
          className="inline-flex h-8 items-center gap-1.5 rounded-[10px] bg-transparent px-3 text-[12px] font-medium text-foreground transition-colors hover:bg-foreground/[0.04] dark:hover:bg-white/[0.05]"
        >
          <HugeiconsIcon
            icon={RefreshIcon}
            strokeWidth={1.75}
            className="size-3.5"
          />
          Keep searching
        </button>
      </div>
    </div>
  );
}

export function InventoryErrorState({
  isDataset,
  onRetry,
}: {
  isDataset: boolean;
  onRetry: () => void;
}) {
  return (
    <div className="flex min-h-[260px] flex-col items-center justify-center gap-3 px-6 text-center">
      <div className="inline-flex size-11 items-center justify-center rounded-[12px] bg-amber-500/10 text-amber-700 dark:text-amber-300">
        <HugeiconsIcon icon={CloudOffIcon} strokeWidth={1.6} className="size-5" />
      </div>
      <div className="space-y-1">
        <p className="text-[14px] font-semibold tracking-tight text-foreground">
          Couldn't load your library
        </p>
        <p className="max-w-md text-[12.5px] leading-5 text-muted-foreground">
          Something went wrong reading your downloaded{" "}
          {isDataset ? "datasets" : "models"}. Check that the backend is running
          and try again.
        </p>
      </div>
      <button
        type="button"
        onClick={onRetry}
        className="inline-flex h-8 items-center gap-1.5 rounded-[10px] bg-transparent px-3 text-[12px] font-medium text-foreground transition-colors hover:bg-foreground/[0.04] dark:hover:bg-white/[0.05]"
      >
        <HugeiconsIcon icon={RefreshIcon} strokeWidth={1.75} className="size-3.5" />
        Try again
      </button>
    </div>
  );
}

export function EmptyState({
  title,
  body,
  icon = CubeIcon,
}: {
  title: string;
  body: string;
  icon?: IconSvgElement;
}) {
  return (
    <div className="flex min-h-[220px] flex-col items-center justify-center gap-3 px-6 text-center">
      <div className="inline-flex size-11 items-center justify-center rounded-[12px] bg-muted text-muted-foreground">
        <HugeiconsIcon icon={icon} strokeWidth={1.5} className="size-5" />
      </div>
      <div className="space-y-1">
        <p className="text-[14px] font-semibold tracking-tight text-foreground">
          {title}
        </p>
        <p className="max-w-md text-[12.5px] leading-5 text-muted-foreground">
          {body}
        </p>
      </div>
    </div>
  );
}

function SkeletonRow() {
  return (
    <div className="flex items-center gap-3 px-3 py-2.5">
      <div className="size-8 shrink-0 animate-pulse rounded-[9px] bg-muted" />
      <div className="min-w-0 flex-1 space-y-1.5">
        <div className="h-[13px] w-1/2 animate-pulse rounded-full bg-muted" />
        <div className="h-[11px] w-3/4 animate-pulse rounded-full bg-muted/70" />
      </div>
    </div>
  );
}

export function SkeletonList({ count = 6 }: { count?: number }) {
  return (
    <ul className="divide-y divide-border" aria-hidden="true">
      {Array.from({ length: count }).map((_, i) => (
        <li key={i}>
          <SkeletonRow />
        </li>
      ))}
    </ul>
  );
}
