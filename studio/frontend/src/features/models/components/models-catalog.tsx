// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Spinner } from "@/components/ui/spinner";
import { cn, formatCompact } from "@/lib/utils";
import {
  CheckmarkCircle02Icon,
  CloudOffIcon,
  CubeIcon,
  DownloadCircle02Icon,
  FavouriteIcon,
  FolderSearchIcon,
  PackageIcon,
  RefreshIcon,
  WifiDisconnected02Icon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { HugeiconsIcon } from "@hugeicons/react";
import type { ReactNode, RefObject } from "react";
import { OwnerAvatar } from "./owner-avatar";
import { formatBytes, formatRelativeShort } from "../lib/format";
import { formatLocalUpdated } from "../lib/view-models";

function StatChip({
  icon,
  value,
}: {
  icon: IconSvgElement;
  value: string;
}) {
  return (
    <span className="inline-flex h-[18px] items-center gap-0.5 rounded-[5px] border border-border/60 bg-background/60 px-1 text-[9.5px] font-medium tabular-nums leading-none text-foreground/80">
      <HugeiconsIcon
        icon={icon}
        strokeWidth={1.75}
        className="size-2.5 text-muted-foreground"
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
  onClick,
  children,
}: {
  selected: boolean;
  active?: boolean;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "group relative block w-full px-3 py-2 text-left transition-colors",
        selected
          ? "bg-foreground/[0.08] hover:bg-foreground/[0.10] dark:bg-white/[0.07] dark:hover:bg-white/[0.09]"
          : "hover:bg-foreground/[0.05] dark:hover:bg-white/[0.04]",
      )}
    >
      {active && (
        <span
          aria-hidden="true"
          className="absolute left-0 top-2 bottom-2 w-[3px] rounded-r-full bg-emerald-500"
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
        className="inline-flex h-8 items-center gap-1.5 rounded-[8px] border border-border/60 bg-muted/40 px-3 text-[12px] font-medium text-foreground transition-colors hover:bg-muted"
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
}: {
  title: string;
  body: string;
}) {
  return (
    <div className="flex min-h-[220px] flex-col items-center justify-center gap-3 px-6 text-center">
      <div className="inline-flex size-11 items-center justify-center rounded-[12px] bg-muted text-muted-foreground">
        <HugeiconsIcon icon={CubeIcon} strokeWidth={1.5} className="size-5" />
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
      <span className="rounded-[6px] bg-muted px-1.5 py-0.5 text-[10.5px] font-medium text-muted-foreground">
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
  return (
    <CatalogRow selected={selected} active={active} onClick={onClick}>
      <div className="flex items-center gap-2.5">
        <OwnerAvatar owner={row.owner} className="size-9 rounded-[10px]" />
        <div className="min-w-0 flex-1">
          <div className="flex h-[18px] min-w-0 items-center justify-between gap-2">
            <div className="flex min-w-0 items-center gap-1.5">
              <p className="truncate text-[13px] font-medium leading-[18px] tracking-tight text-foreground">
                {row.repo}
              </p>
              {row.result.isGguf && (
                <span className="inline-flex h-[14px] shrink-0 items-center rounded-full border border-blue-500/40 px-1 text-[8px] font-semibold uppercase tracking-[0.08em] leading-none text-blue-600 dark:text-blue-400">
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
                icon={DownloadCircle02Icon}
                value={formatCompact(row.result.downloads)}
              />
              <StatChip
                icon={FavouriteIcon}
                value={formatCompact(row.result.likes)}
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
}: {
  row: CachedInventoryRow | LocalInventoryRow;
  selected: boolean;
  activeCheckpoint: string | null;
  onClick: () => void;
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

  return (
    <CatalogRow selected={selected} active={active} onClick={onClick}>
      <div className="flex items-center gap-2.5">
        <OwnerAvatar owner={row.owner} className="size-9 rounded-[10px]" />
        <div className="min-w-0 flex-1">
          <div className="flex h-[18px] min-w-0 items-center justify-between gap-2">
            <div className="flex min-w-0 items-center gap-1.5">
              <p className="truncate text-[13px] font-medium leading-[18px] tracking-tight text-foreground">
                {title}
              </p>
              {row.isGguf && (
                <span className="inline-flex h-[14px] shrink-0 items-center rounded-full border border-blue-500/40 px-1 text-[8px] font-semibold uppercase tracking-[0.08em] leading-none text-blue-600 dark:text-blue-400">
                  GGUF
                </span>
              )}
            </div>
            {row.kind === "cache" && (
              <StatChip icon={PackageIcon} value={formatBytes(row.bytes)} />
            )}
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
}) {
  return (
    <div
      ref={scrollRef}
      className="h-[calc(100dvh-205px)] overflow-y-auto"
    >
      {tab === "discover" ? (
        <>
          {searchError && discoverRows.length === 0 ? (
            <NetworkErrorState
              online={online}
              message={searchError}
              onRetry={onRetry}
            />
          ) : discoverRows.length === 0 ? (
            <EmptyState
              title={
                query.trim()
                  ? "No matching models"
                  : isLoading
                    ? "Loading models"
                    : "No models available"
              }
              body={
                query.trim()
                  ? "Try a broader search or remove some filters."
                  : isLoading
                    ? "The discovery feed is still warming up."
                    : "The current filters are excluding every result."
              }
            />
          ) : (
            <ul className="divide-y divide-border">
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
          title="Nothing on device yet"
          body={
            query.trim()
              ? "No cached or local models match this search."
              : "Downloaded repositories and indexed local folders will appear here."
          }
        />
      ) : (
        <div className="space-y-3">
          {cachedRows.length > 0 && (
            <section>
              <SectionLabel
                icon={PackageIcon}
                label="Hub cache"
                count={cachedRows.length}
              />
              <ul className="divide-y divide-border">
                {cachedRows.map((row) => (
                  <li key={row.id}>
                    <InventoryRow
                      row={row}
                      selected={selectedId === row.id}
                      activeCheckpoint={activeCheckpoint}
                      onClick={() => onSelect(row.id)}
                    />
                  </li>
                ))}
              </ul>
            </section>
          )}

          {localRows.length > 0 && (
            <section>
              <SectionLabel
                icon={FolderSearchIcon}
                label="Local libraries"
                count={localRows.length}
              />
              <ul className="divide-y divide-border">
                {localRows.map((row) => (
                  <li key={row.id}>
                    <InventoryRow
                      row={row}
                      selected={selectedId === row.id}
                      activeCheckpoint={activeCheckpoint}
                      onClick={() => onSelect(row.id)}
                    />
                  </li>
                ))}
              </ul>
            </section>
          )}
        </div>
      )}
    </div>
  );
}
