// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Spinner } from "@/components/ui/spinner";
import { cn, formatCompact } from "@/lib/utils";
import {
  CheckmarkCircle02Icon,
  CloudOffIcon,
  CubeIcon,
  Download01Icon,
  DownloadCircle02Icon,
  FavouriteIcon,
  FolderSearchIcon,
  PackageIcon,
  RefreshIcon,
  WifiDisconnected02Icon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactNode, type RefObject, useEffect, useState } from "react";
import { ModelDeleteAction } from "@/components/assistant-ui/model-selector/model-delete-action";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  deleteCachedModel,
  listGgufVariants,
} from "@/features/chat/api/chat-api";
import type { GgufVariantDetail } from "@/features/chat/types/api";
import { OwnerAvatar } from "./owner-avatar";
import { ACCENT_RULE, type AccentSlug, pickAccent } from "../lib/accent";
import { formatBytes, formatRelativeShort } from "../lib/format";
import { formatLocalUpdated } from "../lib/view-models";

/**
 * Cached GGUF size chip — same look as a StatChip but lazily fetches the
 * per-variant breakdown when the user hovers/opens it. The list of files
 * appears in a `rich` tooltip so users can see exactly which quantizations
 * make up the total size shown on the chip.
 */
function CachedSizeChip({
  repoId,
  totalBytes,
  isGguf,
}: {
  repoId: string;
  totalBytes: number;
  isGguf: boolean;
}) {
  // For GGUF repos, the cache may hold multiple downloaded quants — fetch the
  // per-variant breakdown on mount so the tooltip is ready by the time the
  // user hovers (otherwise it would render twice: once with the total, then
  // again with the file list once the async resolves).
  // For non-GGUF (safetensors) repos, the listing isn't needed — those repos
  // present as a single model artifact, so the tooltip shows one row with the
  // repo id and total size.
  const [variants, setVariants] = useState<GgufVariantDetail[] | null>(null);
  useEffect(() => {
    if (!isGguf) return;
    let canceled = false;
    listGgufVariants(repoId)
      .then((res) => {
        if (canceled) return;
        setVariants(res.variants.filter((v) => v.downloaded));
      })
      .catch(() => {
        if (canceled) return;
        setVariants([]);
      });
    return () => {
      canceled = true;
    };
  }, [repoId, isGguf]);

  const displayBytes =
    isGguf && variants && variants.length > 0
      ? variants.reduce((sum, v) => sum + v.size_bytes, 0)
      : totalBytes;

  const trigger = (
    <span onClick={(e) => e.stopPropagation()}>
      <StatChip icon={PackageIcon} value={formatBytes(displayBytes)} />
    </span>
  );

  // Wait for the GGUF breakdown to land before we wrap with a tooltip — keeps
  // the first hover content stable.
  if (isGguf && (!variants || variants.length === 0)) {
    return trigger;
  }

  const rows: Array<{ filename: string; size_bytes: number }> = isGguf
    ? variants!
    : [{ filename: repoId, size_bytes: totalBytes }];

  return (
    <Tooltip>
      <TooltipTrigger asChild>{trigger}</TooltipTrigger>
      <TooltipContent variant="default" side="top" sideOffset={4}>
        <ul className="flex flex-col gap-1">
          {rows.map((row) => (
            <li
              key={row.filename}
              className="flex items-center gap-3 tabular-nums"
            >
              <span className="min-w-0 truncate">{row.filename}</span>
              <span className="ml-auto">
                <StatChip
                  icon={PackageIcon}
                  value={formatBytes(row.size_bytes)}
                />
              </span>
            </li>
          ))}
        </ul>
      </TooltipContent>
    </Tooltip>
  );
}

function StatChip({
  icon,
  value,
}: {
  icon: IconSvgElement;
  value: string;
}) {
  return (
    <span className="inline-flex shrink-0 items-center gap-1 whitespace-nowrap text-[10.5px] font-medium leading-none tabular-nums text-muted-foreground">
      <HugeiconsIcon
        icon={icon}
        strokeWidth={2}
        className="size-3 shrink-0"
      />
      {value}
    </span>
  );
}
import type {
  CachedInventoryRow,
  DiscoverRow,
  LocalInventoryRow,
  ModelsTab,
} from "../types";

function CatalogRow({
  selected,
  active,
  accent = "slate",
  onClick,
  children,
}: {
  selected: boolean;
  active?: boolean;
  accent?: AccentSlug;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      data-selected={selected || undefined}
      onClick={onClick}
      className="catalog-row group/row relative block w-full cursor-pointer select-none overflow-hidden rounded-[14px] pl-3 pr-2 py-2.5 text-left"
    >
      {active && (
        <span
          aria-hidden="true"
          className={cn(
            "absolute left-0 top-1.5 bottom-1.5 w-[3px] rounded-r-full",
            ACCENT_RULE[accent],
          )}
        />
      )}
      {children}
    </button>
  );
}

function NetworkErrorState({
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

function EmptyState({
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
    <div className="flex items-center gap-2.5 px-3 py-2">
      <div className="size-9 shrink-0 animate-pulse rounded-[10px] bg-muted" />
      <div className="min-w-0 flex-1 space-y-1.5">
        <div className="h-[14px] w-1/2 animate-pulse rounded-full bg-muted" />
        <div className="h-[12px] w-3/4 animate-pulse rounded-full bg-muted/70" />
      </div>
    </div>
  );
}

function SkeletonList({ count = 6 }: { count?: number }) {
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

function SectionLabel({
  icon,
  label,
  count,
}: {
  icon: IconSvgElement;
  label: string;
  count: number;
}) {
  return (
    <div className="mb-1 flex items-center gap-2 px-1 pt-2">
      <HugeiconsIcon
        icon={icon}
        strokeWidth={1.75}
        className="size-3.5 text-muted-foreground"
      />
      <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
        {label}
      </p>
      <span className="text-[10.5px] font-medium tabular-nums text-muted-foreground/70">
        {count}
      </span>
    </div>
  );
}

function DiscoverModelRow({
  row,
  selected,
  active,
  onClick,
}: {
  row: DiscoverRow;
  selected: boolean;
  active: boolean;
  onClick: () => void;
}) {
  const accent = pickAccent(row.capabilities);
  return (
    <CatalogRow
      selected={selected}
      active={active}
      accent={accent}
      onClick={onClick}
    >
      <div className="flex items-center gap-2.5">
        <OwnerAvatar owner={row.owner} className="size-9 rounded-[10px]" />
        <div className="min-w-0 flex-1">
          <div className="flex h-[18px] min-w-0 items-center justify-between gap-2">
            <div className="flex min-w-0 items-center gap-1.5">
              <p className="truncate text-[13px] font-medium leading-[18px] tracking-tight text-foreground">
                {row.repo}
              </p>
              {row.result.isGguf && (
                <span className="shrink-0 text-[9.5px] font-semibold uppercase tracking-[0.08em] leading-none text-blue-600 dark:text-blue-400">
                  GGUF
                </span>
              )}
              {row.isAvailableOnDevice && (
                <HugeiconsIcon
                  icon={CheckmarkCircle02Icon}
                  strokeWidth={2}
                  className="size-3 shrink-0 text-emerald-600 dark:text-emerald-400"
                />
              )}
            </div>
            <div className="flex shrink-0 items-center gap-1">
              <StatChip
                icon={FavouriteIcon}
                value={formatCompact(row.result.likes)}
              />
              <StatChip
                icon={Download01Icon}
                value={formatCompact(row.result.downloads)}
              />
            </div>
          </div>
          <div className="flex h-[18px] min-w-0 items-center justify-between gap-2 text-[12px] leading-[18px] text-muted-foreground">
            <span className="truncate">{row.owner}</span>
            <span className="shrink-0 tabular-nums">
              {formatRelativeShort(row.result.updatedAt)}
            </span>
          </div>
        </div>
      </div>
    </CatalogRow>
  );
}

function cachedRowActive(
  row: CachedInventoryRow,
  activeCheckpoint: string | null,
): boolean {
  return activeCheckpoint?.toLowerCase() === row.repoId.toLowerCase();
}

function localRowActive(
  row: LocalInventoryRow,
  activeCheckpoint: string | null,
): boolean {
  return activeCheckpoint?.toLowerCase() === row.id.toLowerCase();
}

function InventoryRow({
  row,
  selected,
  activeCheckpoint,
  onClick,
  onChange,
}: {
  row: CachedInventoryRow | LocalInventoryRow;
  selected: boolean;
  activeCheckpoint: string | null;
  onClick: () => void;
  onChange?: () => void;
}) {
  const active =
    row.kind === "cache"
      ? cachedRowActive(row, activeCheckpoint)
      : localRowActive(row, activeCheckpoint);
  const title = row.kind === "cache" ? row.repo : row.title;

  const subLabel = row.kind === "cache" ? row.owner : row.sourceLabel;
  const trailing =
    row.kind === "local" && row.updatedAt
      ? formatLocalUpdated(row.updatedAt)
      : null;
  const canDelete = row.kind === "cache";
  const accent = pickAccent(undefined);

  return (
    <CatalogRow
      selected={selected}
      active={active}
      accent={accent}
      onClick={onClick}
    >
      <div className="group/inventory flex items-center gap-2.5">
        <OwnerAvatar owner={row.owner} className="size-9 rounded-[10px]" />
        <div className="min-w-0 flex-1">
          <div className="flex h-[18px] min-w-0 items-center justify-between gap-2">
            <div className="flex min-w-0 items-center gap-1.5">
              <p className="truncate text-[13px] font-medium leading-[18px] tracking-tight text-foreground">
                {title}
              </p>
              {row.isGguf && (
                <span className="shrink-0 text-[9.5px] font-semibold uppercase tracking-[0.08em] leading-none text-blue-600 dark:text-blue-400">
                  GGUF
                </span>
              )}
            </div>
            <div className="flex shrink-0 items-center gap-1">
              {canDelete && (
                <ModelDeleteAction
                  ariaLabel={`Delete ${row.repoId}`}
                  title="Delete cached model?"
                  description={
                    <>
                      This will remove{" "}
                      <span className="font-medium text-foreground">
                        {row.repoId}
                      </span>{" "}
                      {row.isGguf
                        ? "and all of its downloaded quantizations"
                        : "and all of its downloaded files"}{" "}
                      ({formatBytes(row.bytes)}) from disk. You can re-download
                      it later.
                    </>
                  }
                  successMessage={`Deleted ${row.repoId}`}
                  buttonClassName="opacity-0 transition-opacity group-hover/inventory:opacity-100 focus-visible:opacity-100 data-[state=open]:opacity-100"
                  iconClassName="size-3.5"
                  onConfirm={() => deleteCachedModel(row.repoId)}
                  onDeleted={onChange}
                />
              )}
              {row.kind === "cache" && (
                <CachedSizeChip
                  repoId={row.repoId}
                  totalBytes={row.bytes}
                  isGguf={row.isGguf}
                />
              )}
            </div>
          </div>
          <div className="flex h-[18px] min-w-0 items-center justify-between gap-2 text-[12px] leading-[18px] text-muted-foreground">
            <span className="truncate">{subLabel}</span>
            {trailing && (
              <span className="shrink-0 tabular-nums">{trailing}</span>
            )}
          </div>
        </div>
      </div>
    </CatalogRow>
  );
}

export function ModelsCatalog({
  tab,
  discoverRows,
  cachedRows,
  localRows,
  selectedId,
  onSelect,
  isLoading,
  isLoadingMore,
  downloadedReady,
  query,
  scrollRef,
  sentinelRef,
  activeCheckpoint,
  searchError,
  online,
  onRetry,
  onInventoryChange,
}: {
  tab: ModelsTab;
  discoverRows: DiscoverRow[];
  cachedRows: CachedInventoryRow[];
  localRows: LocalInventoryRow[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  isLoading: boolean;
  isLoadingMore: boolean;
  downloadedReady: boolean;
  query: string;
  scrollRef: RefObject<HTMLDivElement | null>;
  sentinelRef: RefObject<HTMLDivElement | null>;
  activeCheckpoint: string | null;
  searchError: string | null;
  online: boolean;
  onRetry: () => void;
  onInventoryChange?: () => void;
}) {
  const [scrolled, setScrolled] = useState(false);
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const onScroll = () => setScrolled(el.scrollTop > 0);
    onScroll();
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, [scrollRef]);

  return (
    <div
      ref={scrollRef}
      className={cn(
        "h-[calc(100dvh-205px)] overflow-y-auto border-t border-transparent pb-6 pl-6 pr-4 pt-4 transition-colors [scrollbar-gutter:stable] [scrollbar-width:thin]",
        scrolled && "border-border",
      )}
    >
      {tab === "discover" ? (
        <>
          {searchError && discoverRows.length === 0 ? (
            <NetworkErrorState
              online={online}
              message={searchError}
              onRetry={onRetry}
            />
          ) : discoverRows.length === 0 && isLoading ? (
            <SkeletonList />
          ) : discoverRows.length === 0 ? (
            <EmptyState
              icon={query.trim() ? FolderSearchIcon : CubeIcon}
              title={
                query.trim() ? "No matching models" : "No models available"
              }
              body={
                query.trim()
                  ? "Try a broader search or remove some filters."
                  : "The current filters are excluding every result."
              }
            />
          ) : (
            <ul>
              {discoverRows.map((row) => (
                <li key={row.id}>
                  <DiscoverModelRow
                    row={row}
                    selected={selectedId === row.id}
                    active={
                      activeCheckpoint?.toLowerCase() === row.id.toLowerCase()
                    }
                    onClick={() => onSelect(row.id)}
                  />
                </li>
              ))}
            </ul>
          )}

          <div ref={sentinelRef} className="h-px" />
          {isLoadingMore && (
            <div className="flex items-center justify-center py-4">
              <Spinner className="size-4 text-muted-foreground" />
            </div>
          )}
        </>
      ) : !downloadedReady ? (
        <div className="flex min-h-[240px] items-center justify-center gap-3 text-[13px] text-muted-foreground">
          <Spinner className="size-4" />
          Loading local inventory…
        </div>
      ) : cachedRows.length === 0 && localRows.length === 0 ? (
        <EmptyState
          icon={query.trim() ? FolderSearchIcon : DownloadCircle02Icon}
          title={
            query.trim() ? "No matches on device" : "Nothing on device yet"
          }
          body={
            query.trim()
              ? "Clear the search or try a different query. No cached or local model matches it."
              : "Downloaded repositories and indexed local folders will appear here."
          }
        />
      ) : (
        <ul>
          {cachedRows.map((row) => (
            <li key={row.id}>
              <InventoryRow
                row={row}
                selected={selectedId === row.id}
                activeCheckpoint={activeCheckpoint}
                onClick={() => onSelect(row.id)}
                onChange={onInventoryChange}
              />
            </li>
          ))}
          {localRows.map((row) => (
            <li key={row.id}>
              <InventoryRow
                row={row}
                selected={selectedId === row.id}
                activeCheckpoint={activeCheckpoint}
                onClick={() => onSelect(row.id)}
                onChange={onInventoryChange}
              />
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
