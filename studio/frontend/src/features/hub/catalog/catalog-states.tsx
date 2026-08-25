// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  CloudOffIcon,
  CubeIcon,
  FilterIcon,
  Refresh01Icon,
  WifiDisconnected02Icon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { HugeiconsIcon } from "@hugeicons/react";
import type { ReactNode } from "react";
import { useLayoutEffect, useRef, useState } from "react";
import type { HubFailure } from "@/features/hub/lib/network";

// Only a browser reporting itself offline earns "You're offline". Calling a DNS
// filter or extension block "offline" is what made these bugs undiagnosable.
function describeFailure(
  failure: HubFailure | null | undefined,
  online: boolean,
  resourceLabel: "models" | "datasets",
): { title: string; body: string; offlineLike: boolean } {
  switch (failure?.kind) {
    case "browser-offline":
      return {
        title: "You're offline",
        body: `Reconnect to the internet to browse ${resourceLabel} from Hugging Face.`,
        offlineLike: true,
      };
    case "timeout":
      return {
        title: "Hugging Face timed out",
        body: failure.message,
        offlineLike: false,
      };
    case "network-opaque":
    case "unknown":
      return {
        title: "Can't reach Hugging Face",
        body: failure.message,
        offlineLike: false,
      };
    default:
      break;
  }
  return online
    ? {
        title: "Couldn't reach Hugging Face",
        body: "The discovery feed couldn't load. Check your connection or try again.",
        offlineLike: false,
      }
    : {
        title: "Can't reach Hugging Face",
        body: `Studio couldn't load ${resourceLabel} from Hugging Face.`,
        offlineLike: false,
      };
}

export function NetworkErrorState({
  online,
  message,
  failure,
  onRetry,
  onSwitchDevice,
  resourceLabel = "models",
}: {
  online: boolean;
  message: string;
  failure?: HubFailure | null;
  onRetry: () => void;
  onSwitchDevice?: () => void;
  resourceLabel?: "models" | "datasets";
}) {
  const { title, body, offlineLike } = describeFailure(
    failure,
    online,
    resourceLabel,
  );
  const icon = offlineLike ? WifiDisconnected02Icon : CloudOffIcon;

  return (
    <div className="flex min-h-[260px] flex-col items-center justify-center gap-3 px-6 text-center">
      <div className="inline-flex size-11 items-center justify-center rounded-[12px] bg-amber-500/10 text-amber-700 dark:text-amber-300">
        <HugeiconsIcon icon={icon} strokeWidth={1.6} className="size-5" />
      </div>
      <div className="space-y-1">
        <p className="text-ui-14 font-semibold tracking-tight text-foreground">
          {title}
        </p>
        <p className="max-w-md text-ui-12p5 leading-5 text-muted-foreground">
          {body}
        </p>
        <p className="text-ui-11 text-muted-foreground/70">{message}</p>
      </div>
      <div className="flex flex-wrap items-center justify-center gap-2">
        {onSwitchDevice ? (
          <button
            type="button"
            onClick={onSwitchDevice}
            className="inline-flex h-8 items-center gap-1.5 rounded-full bg-foreground/[0.06] px-3 text-ui-12 font-medium text-foreground transition-colors hover:bg-foreground/[0.1] dark:bg-white/[0.06] dark:hover:bg-white/[0.1]"
          >
            On Device
          </button>
        ) : null}
        <button
          type="button"
          onClick={onRetry}
          className="inline-flex h-8 items-center gap-1.5 rounded-full bg-transparent px-3 text-ui-12 font-medium text-foreground transition-colors hover:bg-foreground/[0.04] dark:hover:bg-white/[0.05]"
        >
          <HugeiconsIcon
            icon={Refresh01Icon}
            strokeWidth={1.75}
            className="size-3.5"
          />
          Try again
        </button>
      </div>
    </div>
  );
}

export function DiscoverFetchMoreState({
  scannedCount,
  hasActiveFilters,
  isLoadingMore,
  onFetchMore,
  onClearFilters,
}: {
  scannedCount: number;
  hasActiveFilters: boolean;
  isLoadingMore: boolean;
  onFetchMore: () => void;
  onClearFilters: () => void;
}) {
  return (
    <div className="flex min-h-[260px] flex-col items-center justify-center gap-3 px-6 text-center">
      <div className="inline-flex size-11 items-center justify-center rounded-[12px] bg-muted text-muted-foreground">
        <HugeiconsIcon icon={FilterIcon} strokeWidth={1.5} className="size-5" />
      </div>
      <div className="space-y-1">
        <p className="text-ui-14 font-semibold tracking-tight text-foreground">
          No matches yet
        </p>
        <p className="max-w-md text-ui-12p5 leading-5 text-muted-foreground">
          Scanned {scannedCount.toLocaleString()} results. Load another page to
          keep searching Hugging Face.
        </p>
      </div>
      <div className="flex flex-wrap items-center justify-center gap-2 pt-1">
        {hasActiveFilters && (
          <button
            type="button"
            onClick={onClearFilters}
            className="inline-flex h-8 items-center gap-1.5 rounded-full bg-foreground/[0.06] px-3 text-ui-12 font-medium text-foreground transition-colors hover:bg-foreground/[0.1] dark:bg-white/[0.06] dark:hover:bg-white/[0.1]"
          >
            Clear filters
          </button>
        )}
        <button
          type="button"
          onClick={onFetchMore}
          disabled={isLoadingMore}
          className="inline-flex h-8 items-center gap-1.5 rounded-full bg-transparent px-3 text-ui-12 font-medium text-foreground transition-colors hover:bg-foreground/[0.04] disabled:cursor-not-allowed disabled:opacity-50 dark:hover:bg-white/[0.05]"
        >
          <HugeiconsIcon
            icon={Refresh01Icon}
            strokeWidth={1.75}
            className="size-3.5"
          />
          {isLoadingMore ? "Loading..." : "Load more"}
        </button>
      </div>
    </div>
  );
}

export function DiscoverFetchMoreFooter({
  hasActiveFilters,
  isLoadingMore,
  onFetchMore,
  failed = false,
  failureText,
  onRetry,
}: {
  hasActiveFilters: boolean;
  isLoadingMore: boolean;
  onFetchMore: () => void;
  /** The last attempt failed, so this is the only recovery left on screen. */
  failed?: boolean;
  /** The classified, already sanitized cause. Shown here because this footer
   *  outlives the toast that would otherwise be the only place it appeared. */
  failureText?: string;
  onRetry?: () => void;
}) {
  return (
    <div className="relative z-10 flex flex-col items-center gap-2 rounded-[16px] bg-card px-4 py-4 text-center">
      {/* Only warn about hidden results when a filter is actually narrowing them. */}
      {hasActiveFilters && (
        <p className="text-ui-11p5 leading-4 text-muted-foreground">
          Some results may be hidden by your filters.
        </p>
      )}
      {/* Rows stay on screen when the feed fails, so without this the outage is
          invisible and there is nothing left to click once the toast goes. The
          cause goes here too: naming it is the whole point, and the toast is
          transient, so reducing this to "out of date" threw it away again. */}
      {failed && (
        <p className="max-w-md text-ui-11p5 leading-4 text-muted-foreground">
          {failureText || "These results may be out of date."}
        </p>
      )}
      <button
        type="button"
        onClick={failed && onRetry ? onRetry : onFetchMore}
        disabled={isLoadingMore}
        className="inline-flex h-8 items-center gap-1.5 rounded-full bg-foreground/[0.06] px-3 text-ui-12 font-medium text-foreground transition-colors hover:bg-foreground/[0.1] disabled:cursor-not-allowed disabled:opacity-50 dark:bg-white/[0.06] dark:hover:bg-white/[0.1]"
      >
        <HugeiconsIcon
          icon={Refresh01Icon}
          strokeWidth={1.75}
          className="size-3.5"
        />
        {isLoadingMore ? "Loading..." : failed ? "Try again" : "Load more"}
      </button>
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
        <p className="text-ui-14 font-semibold tracking-tight text-foreground">
          Couldn't load your library
        </p>
        <p className="max-w-md text-ui-12p5 leading-5 text-muted-foreground">
          Something went wrong reading your downloaded{" "}
          {isDataset ? "datasets" : "models"}. Check that the backend is running
          and try again.
        </p>
      </div>
      <button
        type="button"
        onClick={onRetry}
        className="inline-flex h-8 items-center gap-1.5 rounded-full bg-transparent px-3 text-ui-12 font-medium text-foreground transition-colors hover:bg-foreground/[0.04] dark:hover:bg-white/[0.05]"
      >
        <HugeiconsIcon icon={Refresh01Icon} strokeWidth={1.75} className="size-3.5" />
        Try again
      </button>
    </div>
  );
}

export function EmptyState({
  title,
  body,
  icon = CubeIcon,
  action,
}: {
  title: string;
  body: string;
  icon?: IconSvgElement;
  action?: ReactNode;
}) {
  return (
    <div className="flex min-h-[220px] flex-col items-center justify-center gap-3 px-6 text-center">
      <div className="inline-flex size-11 items-center justify-center rounded-[12px] bg-muted text-muted-foreground">
        <HugeiconsIcon icon={icon} strokeWidth={1.5} className="size-5" />
      </div>
      <div className="space-y-1">
        <p className="text-ui-14 font-semibold tracking-tight text-foreground">
          {title}
        </p>
        <p className="max-w-md text-ui-12p5 leading-5 text-muted-foreground">
          {body}
        </p>
      </div>
      {action}
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

const SKELETON_ROW_ESTIMATE_PX = 56;
const MIN_SKELETON_ROWS = 4;
const MAX_SKELETON_ROWS = 24;
const DEFAULT_SKELETON_ROWS = 6;

function clampSkeletonCount(height: number): number {
  if (!Number.isFinite(height) || height <= 0) return DEFAULT_SKELETON_ROWS;
  return Math.max(
    MIN_SKELETON_ROWS,
    Math.min(MAX_SKELETON_ROWS, Math.ceil(height / SKELETON_ROW_ESTIMATE_PX)),
  );
}

export function SkeletonList({ count }: { count?: number }) {
  const ref = useRef<HTMLUListElement>(null);
  const [autoCount, setAutoCount] = useState(count ?? DEFAULT_SKELETON_ROWS);
  const rowCount = count ?? autoCount;

  useLayoutEffect(() => {
    if (count != null) return;
    const container = ref.current?.parentElement;
    if (!container || typeof window === "undefined") return;

    let frame: number | null = null;
    const update = () => {
      frame = null;
      setAutoCount(clampSkeletonCount(container.clientHeight));
    };
    const schedule = () => {
      if (frame !== null) return;
      frame = window.requestAnimationFrame(update);
    };
    schedule();

    if (typeof ResizeObserver === "undefined") {
      window.addEventListener("resize", schedule);
      return () => {
        if (frame !== null) window.cancelAnimationFrame(frame);
        window.removeEventListener("resize", schedule);
      };
    }

    const observer = new ResizeObserver(schedule);
    observer.observe(container);
    return () => {
      if (frame !== null) window.cancelAnimationFrame(frame);
      observer.disconnect();
    };
  }, [count]);

  return (
    <ul ref={ref} className="divide-y divide-border" aria-hidden="true">
      {Array.from({ length: rowCount }).map((_, i) => (
        <li key={i}>
          <SkeletonRow />
        </li>
      ))}
    </ul>
  );
}
