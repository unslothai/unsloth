// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  type GgufVariantDetail,
  deleteCachedDataset,
  deleteCachedModel,
  formatLocalUpdated,
  listGgufVariants,
  useGgufVariantsCacheVersion,
} from "@/features/hub";
import {
  classifyUnslothSupport,
  formatBytes,
  formatRelativeShort,
  ggufVariantDisplayLabel,
  modelIdsMatch,
  useHfTokenStore,
} from "@/features/hub";
import { ModelDeleteAction } from "@/features/model-picker";
import { cn, formatCompact } from "@/lib/utils";
import {
  Download01Icon,
  FavouriteIcon,
  PackageIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { HugeiconsIcon } from "@hugeicons/react";
import { useVirtualizer } from "@tanstack/react-virtual";
import {
  type ReactNode,
  createContext,
  memo,
  useCallback,
  useContext,
  useMemo,
  useRef,
  useState,
} from "react";
import { paramLabelFromId } from "../lib/view-models";
import type {
  CachedInventoryRow,
  DiscoverRow,
  LocalInventoryRow,
} from "../types";
import { OwnerAvatar } from "./owner-avatar";
import { AccessGlyphs } from "./shared";

const COARSE_POINTER =
  typeof window !== "undefined" &&
  typeof window.matchMedia === "function" &&
  window.matchMedia("(pointer: coarse)").matches;

// Defer the cached-size chip (Radix Tooltip + two store subscriptions) until a
// row is first hovered/focused so scrolling the virtualized list doesn't pay
// that cost per row; an identical StatChip placeholder makes the swap invisible.
// Coarse pointers have no hover, so they arm immediately. Default true so any
// out-of-row usage stays functional.
const CatalogRowInteractiveContext = createContext(true);

function CachedSizeChip(props: {
  repoId: string;
  totalBytes: number;
  isGguf: boolean;
  isDataset?: boolean;
  cachePath?: string | null;
}) {
  const interactive = useContext(CatalogRowInteractiveContext);
  if (!interactive) {
    return (
      <StatChip icon={PackageIcon} value={formatBytes(props.totalBytes)} />
    );
  }
  return <CachedSizeChipLive {...props} />;
}

function CachedSizeChipLive({
  repoId,
  totalBytes,
  isGguf,
  isDataset = false,
  cachePath,
}: {
  repoId: string;
  totalBytes: number;
  isGguf: boolean;
  isDataset?: boolean;
  cachePath?: string | null;
}) {
  const hfToken = useHfTokenStore((s) => s.token) || undefined;
  const variantsVersion = useGgufVariantsCacheVersion(repoId);
  const fetchKey = `${repoId}::${cachePath ?? ""}::${variantsVersion}`;
  const [variantState, setVariantState] = useState<{
    key: string;
    status: "idle" | "loading" | "loaded" | "error";
    variants: GgufVariantDetail[];
  }>(() => ({
    key: fetchKey,
    status: "idle",
    variants: [],
  }));
  const fetchedForRef = useRef<string | null>(null);
  const needsVariantFetch = isGguf && !isDataset;
  const currentVariantState =
    variantState.key === fetchKey
      ? variantState
      : { key: fetchKey, status: "idle" as const, variants: [] };

  const ensureVariantsLoaded = useCallback(() => {
    if (!needsVariantFetch) return;
    if (fetchedForRef.current === fetchKey) return;
    fetchedForRef.current = fetchKey;
    setVariantState({ key: fetchKey, status: "loading", variants: [] });
    listGgufVariants(repoId, hfToken, {
      preferLocalCache: true,
      localPath: cachePath ?? null,
    })
      .then((res) => {
        if (fetchedForRef.current !== fetchKey) return;
        setVariantState({
          key: fetchKey,
          status: "loaded",
          variants: res.variants.filter((v) => v.downloaded),
        });
      })
      .catch(() => {
        if (fetchedForRef.current !== fetchKey) return;
        fetchedForRef.current = null;
        setVariantState({ key: fetchKey, status: "error", variants: [] });
      });
  }, [needsVariantFetch, fetchKey, repoId, hfToken, cachePath]);

  const trigger = (
    <span
      className="pointer-events-auto hub-modal-pe-guard"
      onClick={(e) => e.stopPropagation()}
    >
      <StatChip icon={PackageIcon} value={formatBytes(totalBytes)} />
    </span>
  );

  const rows: Array<{ label: string; size_bytes: number }> | null =
    needsVariantFetch
      ? currentVariantState.status === "loaded" &&
          currentVariantState.variants.length > 0
        ? currentVariantState.variants.map((variant) => ({
            label: ggufVariantDisplayLabel(variant),
            size_bytes: variant.size_bytes,
          }))
        : null
      : [{ label: repoId, size_bytes: totalBytes }];
  const variantMessage =
    currentVariantState.status === "loading"
      ? "Loading downloaded variants..."
      : currentVariantState.status === "error"
        ? "Couldn't load downloaded variants. Open again to retry."
        : currentVariantState.status === "loaded"
          ? "No downloaded GGUF variants found."
          : "Open to load downloaded variants.";

  return (
    <Tooltip
      onOpenChange={(open) => {
        if (open) ensureVariantsLoaded();
      }}
    >
      <TooltipTrigger asChild={true}>{trigger}</TooltipTrigger>
      <TooltipContent variant="default" side="top" sideOffset={4}>
        {rows ? (
          <ul className="flex flex-col gap-1">
            {rows.map((row) => (
              <li
                key={row.label}
                className="flex items-center gap-3 tabular-nums"
              >
                <span className="min-w-0 truncate">{row.label}</span>
                <span className="ml-auto">
                  {/* Brightened for the dark tooltip: muted grey reads poorly there. */}
                  <StatChip
                    icon={PackageIcon}
                    value={formatBytes(row.size_bytes)}
                    className="text-[11px] text-white/70"
                  />
                </span>
              </li>
            ))}
          </ul>
        ) : (
          <span className="block max-w-52 text-[11px] leading-4 text-muted-foreground">
            {variantMessage}
          </span>
        )}
      </TooltipContent>
    </Tooltip>
  );
}

// Thin "·" separator for inline meta lines (owner · format · params).
function MetaDivider() {
  return (
    <span aria-hidden="true" className="shrink-0 text-muted-foreground/35">
      ·
    </span>
  );
}

export function StatChip({
  icon,
  value,
  className,
}: {
  icon: IconSvgElement;
  value: string;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex shrink-0 items-center gap-1 whitespace-nowrap text-[10px] font-medium leading-none tabular-nums text-muted-foreground/75",
        className,
      )}
    >
      <HugeiconsIcon
        icon={icon}
        strokeWidth={1.75}
        className="size-2.5 shrink-0"
      />
      {value}
    </span>
  );
}

function CatalogRow({
  selected,
  active,
  onClick,
  tooltip,
  label,
  children,
  variant = "flat",
}: {
  selected: boolean;
  active?: boolean;
  tooltip?: ReactNode;
  onClick: () => void;
  label: string;
  children: ReactNode;
  variant?: "flat" | "card";
}) {
  const [interactive, setInteractive] = useState(COARSE_POINTER);
  const arm = useCallback(() => setInteractive(true), []);
  const card = variant === "card";
  const button = (
    <div
      data-selected={selected || undefined}
      data-active={active || undefined}
      onPointerEnter={arm}
      onFocusCapture={arm}
      className={cn(
        "group/row relative w-full select-none overflow-hidden text-left",
        card
          ? "hub-result-row flex h-full items-center rounded-[16px] px-4"
          : "catalog-row block rounded-[14px] pl-3 pr-2.5 py-2.5",
      )}
    >
      <button
        type="button"
        aria-label={label}
        onClick={onClick}
        className={cn(
          "absolute inset-0 cursor-pointer outline-none focus-visible:ring-1 focus-visible:ring-ring focus-visible:ring-inset",
          card ? "rounded-[16px]" : "rounded-[14px]",
        )}
      />
      <CatalogRowInteractiveContext.Provider value={interactive}>
        <div
          className={cn("pointer-events-none relative", card && "z-[1] w-full")}
        >
          {children}
        </div>
      </CatalogRowInteractiveContext.Provider>
    </div>
  );
  if (!tooltip || !interactive) return button;
  return (
    <Tooltip>
      <TooltipTrigger asChild={true}>{button}</TooltipTrigger>
      <TooltipContent
        side={card ? "top" : "right"}
        align="start"
        sideOffset={card ? 6 : 8}
        className="tooltip-compact max-w-xs"
      >
        {tooltip}
      </TooltipContent>
    </Tooltip>
  );
}

function StatusDot({
  tone,
  label,
}: {
  tone: "warning" | "danger" | "success";
  label: string;
}) {
  const toneClass =
    tone === "warning"
      ? "bg-status-warning"
      : tone === "danger"
        ? "bg-status-danger"
        : "bg-status-success";
  return (
    <span
      role="img"
      aria-label={label}
      className={cn("inline-block size-[5px] shrink-0 rounded-full", toneClass)}
    />
  );
}

function TooltipLegendRow({
  toneClass,
  children,
}: {
  toneClass: string;
  children: ReactNode;
}) {
  return (
    <div className="flex items-start gap-1.5 leading-snug">
      <span
        aria-hidden="true"
        className={cn(
          "mt-1 inline-block size-1.5 shrink-0 rounded-full",
          toneClass,
        )}
      />
      <span className="min-w-0">{children}</span>
    </div>
  );
}

export function buildRowStatusTooltip({
  isGguf,
  isAdapter,
  isAvailableOnDevice,
  partialRepoId,
  unsupported,
  unsupportedReason,
  resourceLabel = "model",
}: {
  isGguf?: boolean;
  isAdapter?: boolean;
  isAvailableOnDevice?: boolean;
  partialRepoId?: string;
  unsupported?: boolean;
  unsupportedReason?: string | null;
  resourceLabel?: "model" | "dataset";
}): ReactNode {
  const lines: ReactNode[] = [];

  if (isGguf) {
    lines.push(
      <TooltipLegendRow key="gguf" toneClass="bg-format-gguf">
        GGUF format.
      </TooltipLegendRow>,
    );
  }
  if (isAdapter) {
    lines.push(
      <TooltipLegendRow key="adapter" toneClass="bg-format-adapter">
        Adapter format.
      </TooltipLegendRow>,
    );
  }

  if (partialRepoId) {
    lines.push(
      <TooltipLegendRow key="partial" toneClass="bg-status-warning">
        Partial download of <span className="font-medium">{partialRepoId}</span>
        . Click Resume to continue.
      </TooltipLegendRow>,
    );
  } else if (isAvailableOnDevice) {
    lines.push(
      <TooltipLegendRow key="success" toneClass="bg-status-success">
        On device. Ready to use locally.
      </TooltipLegendRow>,
    );
  }

  if (unsupported) {
    lines.push(
      <TooltipLegendRow key="danger" toneClass="bg-status-danger">
        <span className="block">
          This {resourceLabel} may not be supported yet.
        </span>
        {unsupportedReason && (
          <span className="mt-0.5 block text-white/75">
            {unsupportedReason}
          </span>
        )}
        <span className="mt-0.5 block text-white/75">
          Still downloadable to your Hugging Face cache.
        </span>
      </TooltipLegendRow>,
    );
  }

  if (lines.length === 0) return null;
  return <div className="space-y-1.5">{lines}</div>;
}

export const DiscoverModelRow = memo(function DiscoverModelRow({
  row,
  selected,
  active,
  deviceType,
  isDataset,
  onSelect,
}: {
  row: DiscoverRow;
  selected: boolean;
  active: boolean;
  deviceType: string | null;
  isDataset: boolean;
  onSelect: (id: string) => void;
}) {
  const support = useMemo(
    () =>
      isDataset
        ? null
        : classifyUnslothSupport({
            modelId: row.id,
            pipelineTag: row.result.pipelineTag,
            tags: row.result.tags,
            libraryName: row.result.libraryName,
            deviceType,
            quantMethod: row.result.quantMethod,
          }),
    [isDataset, row.id, row.result, deviceType],
  );
  const unsupported = support?.status === "unsupported";
  const handleClick = useCallback(() => onSelect(row.id), [onSelect, row.id]);
  const partialRepoId =
    row.isAvailableOnDevice && row.isPartialOnDevice
      ? row.result.id
      : undefined;
  const tooltip = buildRowStatusTooltip({
    isGguf: row.result.isGguf,
    isAdapter: false,
    isAvailableOnDevice: row.isAvailableOnDevice,
    partialRepoId,
    unsupported,
    unsupportedReason: support?.reason ?? null,
    resourceLabel: isDataset ? "dataset" : "model",
  });
  return (
    <CatalogRow
      selected={selected}
      active={active}
      tooltip={tooltip}
      label={row.repo}
      onClick={handleClick}
    >
      <div className="flex items-center gap-3">
        <OwnerAvatar
          owner={row.owner}
          repoName={row.repo}
          className="size-8 rounded-[11px]"
          remote={false}
        />
        <div className="flex min-w-0 flex-1 flex-col gap-[3px]">
          <div className="flex h-[18px] min-w-0 items-center justify-between gap-2">
            <div className="flex min-w-0 items-center gap-2 pr-2">
              <p className="truncate text-[12px] font-medium leading-[18px] tracking-[-0.005em] text-foreground">
                {row.repo}
              </p>
              <AccessGlyphs
                gated={row.result.gated}
                isPrivate={row.result.private}
                tooltip={false}
              />
              {row.result.isGguf && (
                <span
                  role="img"
                  aria-label="GGUF"
                  className="inline-block size-[5px] shrink-0 rounded-full bg-format-gguf"
                />
              )}
              {unsupported && (
                <StatusDot tone="danger" label="May not be supported yet" />
              )}
              {row.isAvailableOnDevice && row.isPartialOnDevice && (
                <StatusDot tone="warning" label="Partial download" />
              )}
              {row.isAvailableOnDevice && !row.isPartialOnDevice && (
                <StatusDot tone="success" label="On device" />
              )}
            </div>
            <div className="flex shrink-0 items-center gap-2.5">
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
          <div className="flex h-[16px] min-w-0 items-center justify-between gap-2 text-[11.5px] leading-[16px] text-muted-foreground/85">
            <span className="flex min-w-0 items-center gap-1">
              <span className="truncate">{row.owner}</span>
              {row.owner.toLowerCase() === "unsloth" && (
                <span
                  aria-label="Verified Unsloth"
                  className="hub-verified-badge size-3.5 shrink-0 text-verified"
                />
              )}
            </span>
            <span className="shrink-0 text-[10.5px] tabular-nums">
              {formatRelativeShort(row.result.updatedAt)}
            </span>
          </div>
        </div>
      </div>
    </CatalogRow>
  );
});

function cachedRowActive(
  row: CachedInventoryRow,
  activeCheckpoint: string | null,
  activeGgufVariant: string | null,
): boolean {
  if (!modelIdsMatch(activeCheckpoint, row.loadId)) return false;
  if (row.modelFormat === "gguf") {
    return row.capabilities.requiresVariant ? activeGgufVariant !== null : true;
  }
  return activeGgufVariant === null;
}

function localRowActive(
  row: LocalInventoryRow,
  activeCheckpoint: string | null,
  activeGgufVariant: string | null,
): boolean {
  if (!modelIdsMatch(activeCheckpoint, row.loadId)) return false;
  if (row.modelFormat === "gguf") {
    return row.capabilities.requiresVariant ? activeGgufVariant !== null : true;
  }
  return activeGgufVariant === null;
}

export const InventoryRow = memo(function InventoryRow({
  row,
  selected,
  activeCheckpoint,
  activeGgufVariant,
  isDataset,
  dimmed,
  deviceType,
  compact = false,
  onSelect,
  onChange,
}: {
  row: CachedInventoryRow | LocalInventoryRow;
  selected: boolean;
  activeCheckpoint: string | null;
  activeGgufVariant: string | null;
  isDataset: boolean;
  dimmed: boolean;
  deviceType: string | null;
  /** Narrow split master pane: drop the capability column so the name fits. */
  compact?: boolean;
  onSelect: (id: string) => void;
  onChange?: () => void;
}) {
  const rowModelId =
    row.kind === "cache"
      ? row.repoId
      : (row.repoId ?? row.baseModelHubId ?? row.baseModel ?? row.loadId);
  const rowTagsSignature = row.tags?.join("\u0001") ?? "";
  const unsupported = useMemo(() => {
    if (isDataset) return false;
    return (
      classifyUnslothSupport({
        modelId: rowModelId,
        pipelineTag: row.pipelineTag,
        tags: rowTagsSignature ? rowTagsSignature.split("\u0001") : undefined,
        libraryName: row.libraryName,
        quantMethod: row.quantMethod,
        deviceType,
      }).status === "unsupported"
    );
  }, [
    isDataset,
    rowModelId,
    row.pipelineTag,
    rowTagsSignature,
    row.libraryName,
    row.quantMethod,
    deviceType,
  ]);
  const handleClick = useCallback(() => onSelect(row.id), [onSelect, row.id]);
  const active =
    row.kind === "cache"
      ? cachedRowActive(row, activeCheckpoint, activeGgufVariant)
      : localRowActive(row, activeCheckpoint, activeGgufVariant);
  const title = row.kind === "cache" ? row.repo : row.title;

  const subLabel = row.owner;
  const trailing =
    row.kind === "local" && row.updatedAt
      ? formatLocalUpdated(row.updatedAt)
      : null;
  const cacheDeletableRepoId =
    row.kind === "cache"
      ? row.repoId
      : row.source === "hf_cache" && row.repoId
        ? row.repoId
        : null;
  const canDelete = cacheDeletableRepoId !== null;
  const partialRepoId = row.partial
    ? row.kind === "cache"
      ? row.repoId
      : (row.repoId ?? row.loadId)
    : undefined;
  const tooltip = buildRowStatusTooltip({
    isGguf: row.isGguf,
    isAdapter: row.modelFormat === "adapter",
    isAvailableOnDevice: !partialRepoId,
    partialRepoId,
    unsupported,
    resourceLabel: isDataset ? "dataset" : "model",
  });
  // Always-derivable stats so on-device rows don't show empty placeholder cells:
  // format badge, parameter count from the repo name, and GGUF quant variant.
  const formatLabel =
    row.modelFormat === "gguf"
      ? "GGUF"
      : row.modelFormat === "adapter"
        ? "Adapter"
        : row.modelFormat === "safetensors" || row.modelFormat === "checkpoint"
          ? "Safetensors"
          : null;
  const paramLabel = useMemo(() => paramLabelFromId(title), [title]);
  const quantLabel = row.formatVariant?.trim() || null;

  const metaChips =
    !isDataset && (formatLabel || paramLabel || quantLabel) ? (
      <div className="hidden shrink-0 items-center gap-1.5 sm:flex">
        {/* Format already shows as the status dot, so the pill stays neutral. */}
        {formatLabel && <span className="hub-chip">{formatLabel}</span>}
        {paramLabel && (
          <span className="hub-chip tabular-nums">{paramLabel}</span>
        )}
        {quantLabel && (
          <span className="hub-chip font-mono text-[10.5px] uppercase">
            {quantLabel}
          </span>
        )}
      </div>
    ) : null;

  // On-disk size for cached repos, else local source or last-modified date.
  const sourceLabel = row.kind === "local" ? row.sourceLabel : null;

  const statusMarkers = (
    <>
      {row.isGguf && (
        <span
          role="img"
          aria-label="GGUF"
          className="inline-block size-[5px] shrink-0 rounded-full bg-format-gguf"
        />
      )}
      {row.modelFormat === "adapter" && (
        <span
          role="img"
          aria-label="Adapter"
          className="inline-block size-[5px] shrink-0 rounded-full bg-format-adapter"
        />
      )}
      {partialRepoId ? (
        <StatusDot tone="warning" label="Partial download" />
      ) : (
        <StatusDot tone="success" label="On device" />
      )}
      {unsupported && (
        <StatusDot tone="danger" label="May not be supported yet" />
      )}
    </>
  );

  // Compact rows are all on-device, so the format dots are noise: surface only
  // exceptional states; format + params move to the meta line.
  const compactMarkers =
    partialRepoId || unsupported ? (
      <span className="flex shrink-0 items-center gap-1">
        {partialRepoId && <StatusDot tone="warning" label="Partial download" />}
        {unsupported && (
          <StatusDot tone="danger" label="May not be supported yet" />
        )}
      </span>
    ) : null;

  const ownerLine = (
    <span className="mt-0.5 flex min-w-0 items-center gap-1 text-[11.5px] leading-[15px] text-muted-foreground/80">
      <span className="truncate">{subLabel}</span>
      {subLabel.toLowerCase() === "unsloth" && (
        <span
          aria-label="Verified Unsloth"
          className="hub-verified-badge size-3.5 shrink-0 text-verified"
        />
      )}
    </span>
  );

  const deleteAction =
    canDelete && cacheDeletableRepoId ? (
      <ModelDeleteAction
        ariaLabel={`Delete ${cacheDeletableRepoId}`}
        title={isDataset ? "Delete cached dataset?" : "Delete cached model?"}
        description={
          <>
            This will remove{" "}
            <span className="font-medium text-foreground">
              {cacheDeletableRepoId}
            </span>{" "}
            {isDataset
              ? "and its downloaded files"
              : row.isGguf
                ? "and all of its downloaded quantizations"
                : "and all of its downloaded files"}
            {row.kind === "cache" ? ` (${formatBytes(row.bytes)})` : ""} from
            disk. You can re-download it later.
          </>
        }
        successMessage={`Deleted ${cacheDeletableRepoId}`}
        buttonClassName="pointer-events-auto hub-modal-pe-guard p-2 opacity-0 transition-opacity group-hover/row:opacity-100 focus-visible:opacity-100 data-[state=open]:opacity-100 [@media(pointer:coarse)]:opacity-100"
        iconClassName="size-4"
        onConfirm={async () => {
          if (isDataset) {
            await deleteCachedDataset(cacheDeletableRepoId);
          } else {
            await deleteCachedModel(cacheDeletableRepoId);
          }
        }}
        onDeleted={onChange}
      />
    ) : null;

  // Compact master pane: drop the capability column and collapse size + date
  // into one trailing group so the name keeps the whole middle.
  if (compact) {
    return (
      <CatalogRow
        variant="flat"
        selected={selected}
        active={active}
        tooltip={tooltip}
        label={title}
        onClick={handleClick}
      >
        <div
          className={cn(
            "flex w-full items-center gap-2.5 transition-opacity",
            dimmed && "opacity-25 group-hover/row:opacity-60",
          )}
        >
          <OwnerAvatar
            owner={row.owner}
            repoName={title}
            className="size-8 shrink-0 rounded-[9px] text-[12px]"
            remote={false}
          />
          <div className="min-w-0 flex-1">
            <div className="flex min-w-0 items-center gap-1.5">
              <span className="truncate text-[12.5px] font-semibold leading-[16px] text-foreground">
                {title}
              </span>
              {compactMarkers}
            </div>
            <span className="mt-0.5 flex min-w-0 items-center gap-1.5 text-[10.5px] leading-[14px] text-muted-foreground/75">
              <span className="flex min-w-0 items-center gap-1">
                <span className="truncate">{subLabel}</span>
                {subLabel.toLowerCase() === "unsloth" && (
                  <span
                    aria-label="Verified Unsloth"
                    className="hub-verified-badge size-3 shrink-0 text-verified"
                  />
                )}
              </span>
              {formatLabel && (
                <>
                  <MetaDivider />
                  <span className="shrink-0">{formatLabel}</span>
                </>
              )}
              {paramLabel && (
                <>
                  <MetaDivider />
                  <span className="shrink-0 tabular-nums">{paramLabel}</span>
                </>
              )}
            </span>
          </div>
          <div className="flex shrink-0 items-center gap-2 text-[10.5px] tabular-nums text-muted-foreground/70">
            {row.kind === "cache" ? (
              <CachedSizeChip
                repoId={row.repoId}
                totalBytes={row.bytes}
                isGguf={row.isGguf}
                isDataset={isDataset}
                cachePath={row.cachePath}
              />
            ) : trailing ? (
              <span>{trailing}</span>
            ) : null}
            {deleteAction}
          </div>
        </div>
      </CatalogRow>
    );
  }

  return (
    <CatalogRow
      variant="card"
      selected={selected}
      active={active}
      tooltip={tooltip}
      label={title}
      onClick={handleClick}
    >
      <div
        className={cn(
          "flex w-full items-center gap-3 transition-opacity",
          dimmed && "opacity-25 group-hover/row:opacity-60",
        )}
      >
        <div className="flex min-w-0 flex-1 items-center gap-3">
          <OwnerAvatar
            owner={row.owner}
            repoName={title}
            className="size-9 rounded-[12px]"
            remote={false}
          />
          <div className="min-w-0 flex-1">
            <div className="flex min-w-0 items-center gap-1.5">
              <span className="truncate text-[13.5px] font-semibold leading-[17px] text-foreground">
                {title}
              </span>
              {statusMarkers}
            </div>
            {ownerLine}
          </div>
        </div>

        {metaChips}

        <div className="flex w-[96px] shrink-0 items-center justify-end text-right">
          {row.kind === "cache" ? (
            <CachedSizeChip
              repoId={row.repoId}
              totalBytes={row.bytes}
              isGguf={row.isGguf}
              isDataset={isDataset}
              cachePath={row.cachePath}
            />
          ) : trailing ? (
            <span className="truncate text-[11.5px] tabular-nums text-muted-foreground/70">
              {trailing}
            </span>
          ) : sourceLabel ? (
            <span className="truncate text-[11.5px] text-muted-foreground/55">
              {sourceLabel}
            </span>
          ) : null}
        </div>

        <div className="flex w-9 shrink-0 items-center justify-end">
          {deleteAction}
        </div>
      </div>
    </CatalogRow>
  );
});

export const CATALOG_ROW_HEIGHT_PX = 57;

export function VirtualRows<T>({
  items,
  scrollElement,
  getKey,
  renderRow,
  scrollMargin = 0,
  columns = 1,
  rowHeight = CATALOG_ROW_HEIGHT_PX,
  cellHeight = rowHeight,
  columnGap = 12,
}: {
  items: readonly T[];
  scrollElement: HTMLDivElement | null;
  getKey: (item: T, index: number) => string;
  renderRow: (item: T) => ReactNode;
  scrollMargin?: number;
  columns?: number;
  rowHeight?: number;
  cellHeight?: number;
  columnGap?: number;
}) {
  const lanes = Math.max(1, columns);
  const rowCount = Math.ceil(items.length / lanes);
  // eslint-disable-next-line react-hooks/incompatible-library
  const virtualizer = useVirtualizer({
    count: rowCount,
    getScrollElement: () => scrollElement,
    estimateSize: () => rowHeight,
    overscan: 10,
    scrollMargin,
    getItemKey: (rowIndex) => {
      const item = items[rowIndex * lanes];
      return item ? getKey(item, rowIndex * lanes) : `row-${rowIndex}`;
    },
  });

  return (
    <ul
      style={{
        height: virtualizer.getTotalSize(),
        position: "relative",
        width: "100%",
        overflowAnchor: "none",
      }}
    >
      {virtualizer.getVirtualItems().map((virtualRow) => {
        const startIndex = virtualRow.index * lanes;
        return (
          <li
            key={virtualRow.key}
            data-index={virtualRow.index}
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              width: "100%",
              transform: `translateY(${virtualRow.start - scrollMargin}px)`,
              // Fixed height matching estimateSize (no measureElement ref):
              // dynamic per-row measurement churns virtualizer state and causes
              // visible jumps as new rows arrive.
              height: `${rowHeight}px`,
              contain: "layout",
            }}
          >
            <div
              style={{
                display: "grid",
                gridTemplateColumns: `repeat(${lanes}, minmax(0, 1fr))`,
                columnGap: `${columnGap}px`,
                height: `${cellHeight}px`,
              }}
            >
              {Array.from({ length: lanes }, (_, lane) => {
                const item = items[startIndex + lane];
                if (item === undefined) {
                  return <div key={`empty-${lane}`} />;
                }
                return (
                  <div
                    key={getKey(item, startIndex + lane)}
                    className="min-w-0"
                  >
                    {renderRow(item)}
                  </div>
                );
              })}
            </div>
          </li>
        );
      })}
    </ul>
  );
}
