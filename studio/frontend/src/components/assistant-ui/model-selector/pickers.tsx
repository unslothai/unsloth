// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { FormatDot, type FormatDotTone } from "@/components/format-dot";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { usePlatformStore } from "@/config/env";
import { useRecentRank } from "@/features/chat";
import {
  createActiveModelDownloadRefsSelector,
  type ManagedDownload,
  useDownloadManagerStore,
} from "@/features/download-jobs";
import {
  type CachedInventoryRow,
  type GgufVariantDetail,
  type LocalInventoryRow,
  type LocalSource,
  listGgufVariants,
  useGgufVariantsCacheVersion,
} from "@/features/inventory";
import { useGpuInfo } from "@/hooks";
import { formatBytes } from "@/lib/format";
import { type GgufFitTier, classifyGgufFit, ggufFitTier } from "@/lib/gguf-fit";
import { looksLikeLocalPath } from "@/lib/local-path";
import { hasGgufRepoSuffix } from "@/lib/model-identifiers";
import { matchTokens, tokenizeQuery } from "@/lib/search-text";
import { fingerprintToken } from "@/lib/token-fingerprint";
import { cn } from "@/lib/utils";
import type { VramFitStatus } from "@/lib/vram";
import { useHfTokenStore } from "@/stores/hf-token-store";
import {
  ArrowRight01Icon,
  CubeIcon,
  Search01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import {
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";
import {
  isStudioTrainedModel,
  trainedOutputMeta,
  trainingMethodLabel,
} from "./model-output-labels";
import type {
  LoraModelOption,
  ModelPickTarget,
  ModelSelectorChangeMeta,
} from "./types";
import {
  localModelDisplayName,
  localModelGroupLabel,
  useChatPickerInventory,
} from "./use-chat-picker-inventory";
import {
  cachedInventoryRowCanChat,
  localInventoryRowCanChat,
} from "./chat-picker-compatibility";

function ListLabel({ children }: { children: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-1 px-3 pt-2 pb-1">
      <span className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
        {children}
      </span>
    </div>
  );
}

function InventoryScanNotice({
  error,
  hasRows,
  onRetry,
}: {
  error: string;
  hasRows: boolean;
  onRetry: () => void;
}) {
  return (
    <div
      className={cn(
        "mx-2.5 mb-1 rounded-[8px] border px-3 py-2 text-[12px] text-muted-foreground",
        hasRows
          ? "border-amber-500/30 bg-amber-500/10"
          : "border-destructive/30 bg-destructive/10",
      )}
    >
      <div className="flex items-center justify-between gap-3">
        <span>
          {hasRows
            ? "Some on-device sources couldn't be scanned."
            : "Couldn't scan on-device models."}
        </span>
        <button
          type="button"
          onClick={onRetry}
          className="shrink-0 text-[12px] font-medium text-foreground transition-colors hover:text-primary"
        >
          Retry
        </button>
      </div>
      {!hasRows && <div className="mt-1 break-words text-[11px]">{error}</div>}
    </div>
  );
}

const FORMAT_CHIP_BASE =
  "inline-flex h-[18px] shrink-0 items-center justify-center whitespace-nowrap rounded-[7px] border px-1.5 text-[10.5px] font-medium leading-none tracking-tight";
const FORMAT_CHIP_GGUF =
  "border-format-gguf/40 bg-transparent text-format-gguf";
const FORMAT_CHIP_DEFAULT =
  "border-border/60 bg-transparent text-muted-foreground";

function FormatChip({ meta }: { meta: string }) {
  const ggufMatch = /^GGUF(\s*·\s*(.*))?$/i.exec(meta);
  if (ggufMatch) {
    const rest = ggufMatch[2] ?? "";
    return (
      <span className="inline-flex items-center gap-1.5 shrink-0">
        <span className={cn(FORMAT_CHIP_BASE, FORMAT_CHIP_GGUF)}>GGUF</span>
        {rest && (
          <span className="text-[10px] text-muted-foreground tabular-nums">
            {rest}
          </span>
        )}
      </span>
    );
  }
  return (
    <span className={cn(FORMAT_CHIP_BASE, FORMAT_CHIP_DEFAULT)}>{meta}</span>
  );
}

function ModelLabel({
  label,
  className,
}: {
  label: string;
  className?: string;
}) {
  const slash = label.indexOf("/");
  if (slash <= 0 || slash === label.length - 1) {
    return <span className={cn("truncate", className)}>{label}</span>;
  }
  const owner = label.slice(0, slash + 1);
  const name = label.slice(slash + 1);
  return (
    <span className={cn("truncate", className)}>
      <span className="text-muted-foreground/65">{owner}</span>
      <span className="text-foreground">{name}</span>
    </span>
  );
}

function ModelRow({
  label,
  meta,
  selected,
  onClick,
  vramStatus,
  vramEst,
  gpuGb,
  tooltipText,
  showChevron = true,
  expanded,
  formatTone,
}: {
  label: string;
  meta?: string;
  selected?: boolean;
  onClick: () => void;
  vramStatus?: VramFitStatus | null;
  vramEst?: number;
  gpuGb?: number;
  tooltipText?: ReactNode;
  showChevron?: boolean;
  expanded?: boolean;
  /** When set, renders a Hub-style 5px format dot at the leftmost edge. */
  formatTone?: FormatDotTone | null;
}) {
  const exceeds = vramStatus === "exceeds";
  const showVramTooltip =
    vramEst != null && vramEst > 0 && gpuGb != null && gpuGb > 0;
  const vramTooltipText =
    showVramTooltip && vramStatus
      ? exceeds
        ? `Needs ~${vramEst}GB VRAM (GPU: ${gpuGb}GB)`
        : vramStatus === "tight"
          ? `~${vramEst}GB VRAM (tight fit on ${gpuGb}GB)`
          : `~${vramEst}GB VRAM`
      : null;

  const content = (
    <button
      type="button"
      onClick={onClick}
      data-selected={selected || undefined}
      aria-expanded={expanded || undefined}
      className="picker-row group/row flex w-full items-center gap-2 rounded-[14px] px-3 py-2 text-left text-[13.5px]"
    >
      {formatTone ? <FormatDot tone={formatTone} /> : null}
      <ModelLabel
        label={label}
        className={cn("block min-w-0 flex-1", exceeds && "opacity-60")}
      />
      <span className="ml-auto flex items-center gap-1.5 shrink-0">
        {vramStatus === "exceeds" && (
          <span className="rounded-[6px] !bg-red-50 px-1.5 py-0.5 text-[9px] font-medium !text-red-700 dark:!bg-red-950 dark:!text-red-400">
            OOM
          </span>
        )}
        {vramStatus === "tight" && (
          <span className="text-[9px] font-medium !text-amber-400">TIGHT</span>
        )}
        {meta ? <FormatChip meta={meta} /> : null}
        {showChevron && (
          <HugeiconsIcon
            icon={ArrowRight01Icon}
            strokeWidth={1.75}
            className={cn(
              "size-3.5 shrink-0 text-muted-foreground/50 transition-all group-hover/row:text-foreground",
              expanded ? "rotate-90" : "group-hover/row:translate-x-0.5",
            )}
          />
        )}
      </span>
    </button>
  );

  return (
    <Tooltip>
      <TooltipTrigger asChild={true}>{content}</TooltipTrigger>
      <TooltipContent
        side="right"
        align="start"
        sideOffset={8}
        className="tooltip-compact max-w-xs break-all"
      >
        <span className="block break-words">{label}</span>
        {vramTooltipText ? (
          <span className="mt-1 block text-[10px]">{vramTooltipText}</span>
        ) : null}
        {tooltipText ? (
          <span className="mt-1 block break-words text-[10px] text-muted-foreground">
            {tooltipText}
          </span>
        ) : null}
      </TooltipContent>
    </Tooltip>
  );
}

function isActiveModelDownload(job: ManagedDownload): boolean {
  return (
    job.kind === "model" &&
    (job.state === "running" || job.state === "cancelling")
  );
}

function downloadStatusLabel(job: ManagedDownload): string {
  if (job.state === "cancelling") return "Cancelling";
  if (job.fraction > 0) {
    return `Downloading ${Math.round(Math.min(job.fraction, 0.99) * 100)}%`;
  }
  return "Downloading";
}

function DownloadingModelRow({ jobKey }: { jobKey: string }) {
  const job = useDownloadManagerStore((state) => state.jobs[jobKey]);
  if (!job || !isActiveModelDownload(job)) return null;

  const meta = job.variant ? `GGUF · ${job.variant}` : "Safetensors";
  const bytes =
    job.expectedBytes > 0
      ? `${formatBytes(job.downloadedBytes)} / ${formatBytes(job.expectedBytes)}`
      : job.downloadedBytes > 0
        ? formatBytes(job.downloadedBytes)
        : null;

  return (
    <Tooltip>
      <TooltipTrigger asChild={true}>
        <div className="picker-row flex w-full items-center gap-2 rounded-[14px] px-3 py-2 text-left text-[13.5px]">
          <FormatDot tone={job.variant ? "gguf" : "checkpoint"} />
          <ModelLabel label={job.repoId} className="block min-w-0 flex-1" />
          <span className="ml-auto flex shrink-0 items-center gap-1.5">
            <span className="rounded-[6px] border border-border/60 px-1.5 py-0.5 text-[10px] font-medium text-muted-foreground">
              {downloadStatusLabel(job)}
            </span>
            <FormatChip meta={meta} />
          </span>
        </div>
      </TooltipTrigger>
      <TooltipContent
        side="right"
        align="start"
        sideOffset={8}
        className="tooltip-compact max-w-xs break-all"
      >
        <span className="block break-words">{job.repoId}</span>
        {job.variant ? (
          <span className="mt-1 block break-words text-[10px]">
            Quantization: {job.variant}
          </span>
        ) : null}
        {bytes ? (
          <span className="mt-1 block text-[10px] text-muted-foreground">
            {bytes}
          </span>
        ) : null}
      </TooltipContent>
    </Tooltip>
  );
}

// ── GGUF Variant Expander ────────────────────────────────────

function GgufVariantExpander({
  repoId,
  displayName,
  onPick,
  gpuGb,
  systemRamGb,
  sourceOverride,
  exportType,
  preferLocalCache = false,
  localPath = null,
}: {
  repoId: string;
  displayName: string;
  onPick: (target: ModelPickTarget) => void;
  gpuGb?: number;
  systemRamGb?: number;
  sourceOverride?: ModelSelectorChangeMeta["source"];
  exportType?: ModelSelectorChangeMeta["exportType"];
  preferLocalCache?: boolean;
  localPath?: string | null;
}) {
  const hfToken = useHfTokenStore((s) => s.token) || undefined;
  const variantsVersion = useGgufVariantsCacheVersion(repoId);
  const localVariantPath = localPath?.trim() || null;
  const variantKey = `${repoId}::${fingerprintToken(hfToken)}::${
    preferLocalCache || localVariantPath ? "local" : "remote"
  }::${localVariantPath ?? ""}::${variantsVersion}`;
  const [variantState, setVariantState] = useState<{
    key: string;
    variants: GgufVariantDetail[] | null;
    defaultVariant: string | null;
    hasVision: boolean;
    loading: boolean;
    error: string | null;
  }>(() => ({
    key: variantKey,
    variants: null,
    defaultVariant: null,
    hasVision: false,
    loading: true,
    error: null,
  }));
  const currentVariantState =
    variantState.key === variantKey
      ? variantState
      : {
          key: variantKey,
          variants: null,
          defaultVariant: null,
          hasVision: false,
          loading: true,
          error: null,
        };
  const { variants, defaultVariant, hasVision, loading, error } =
    currentVariantState;

  useEffect(() => {
    let canceled = false;

    listGgufVariants(repoId, hfToken, {
      preferLocalCache,
      localPath: localVariantPath,
    })
      .then((res) => {
        if (canceled) return;
        setVariantState({
          key: variantKey,
          variants: res.variants,
          defaultVariant: res.default_variant,
          hasVision: res.has_vision,
          loading: false,
          error: null,
        });
      })
      .catch((err) => {
        if (canceled) return;
        setVariantState({
          key: variantKey,
          variants: null,
          defaultVariant: null,
          hasVision: false,
          loading: false,
          error: err instanceof Error ? err.message : "Failed to load variants",
        });
      });

    return () => {
      canceled = true;
    };
  }, [repoId, hfToken, variantKey, preferLocalCache, localVariantPath]);

  const isLocalModelPath = looksLikeLocalPath(repoId);

  const handleVariantClick = useCallback(
    (
      quant: string,
      downloaded?: boolean,
      partial?: boolean,
      expectedBytes?: number,
    ) => {
      onPick({
        id: repoId,
        displayName,
        isGguf: true,
        supportsTrustRemoteCode: false,
        meta: {
          source: sourceOverride ?? (isLocalModelPath ? "local" : "hub"),
          isLora: false,
          exportType,
          ggufVariant: quant,
          modelFormat: "gguf",
          isDownloaded: isLocalModelPath ? true : downloaded,
          isPartial: isLocalModelPath ? false : partial === true,
          preferLocalCache,
          localPath: localVariantPath ?? (isLocalModelPath ? repoId : null),
          expectedBytes,
        },
      });
    },
    [
      displayName,
      isLocalModelPath,
      localVariantPath,
      onPick,
      preferLocalCache,
      repoId,
      sourceOverride,
      exportType,
    ],
  );

  const getGgufFit = useCallback(
    (sizeBytes: number): GgufFitTier =>
      ggufFitTier(classifyGgufFit(sizeBytes, { gpuGb, systemRamGb })),
    [gpuGb, systemRamGb],
  );

  const availableVariants = useMemo(
    () => (variants ?? []).filter((v) => v.downloaded || v.partial),
    [variants],
  );
  const downloadedVariants = useMemo(
    () => availableVariants.filter((v) => v.downloaded && !v.partial),
    [availableVariants],
  );
  const recentRank = useRecentRank();

  const effectiveRecommended = useMemo(() => {
    if (downloadedVariants.length === 0) return null;
    if (!gpuGb || gpuGb <= 0) return defaultVariant;
    const defaultV = downloadedVariants.find((v) => v.quant === defaultVariant);
    if (defaultV && getGgufFit(defaultV.size_bytes) !== "oom")
      return defaultVariant;
    const fitting = downloadedVariants.filter(
      (v) => getGgufFit(v.size_bytes) !== "oom",
    );
    if (fitting.length > 0) {
      fitting.sort((a, b) => b.size_bytes - a.size_bytes);
      return fitting[0].quant;
    }
    const sorted = [...downloadedVariants].sort(
      (a, b) => a.size_bytes - b.size_bytes,
    );
    return sorted[0]?.quant ?? null;
  }, [downloadedVariants, defaultVariant, gpuGb, getGgufFit]);

  const sortedVariants = useMemo(() => {
    const tierOf = (v: GgufVariantDetail) => {
      const f = getGgufFit(v.size_bytes);
      if (f === "oom") return 2;
      if (f === "tight") return 1;
      return 0;
    };
    const statusOf = (v: GgufVariantDetail) =>
      v.downloaded && !v.partial ? 0 : v.partial ? 1 : 2;
    return [...availableVariants].sort((a, b) => {
      const aStatus = statusOf(a);
      const bStatus = statusOf(b);
      if (aStatus !== bStatus) return aStatus - bStatus;
      const aTier = tierOf(a);
      const bTier = tierOf(b);
      if (aTier !== bTier) return aTier - bTier;
      const aRecent = recentRank.rank(repoId, a.quant);
      const bRecent = recentRank.rank(repoId, b.quant);
      if (aRecent !== bRecent) {
        if (Number.isFinite(aRecent) && Number.isFinite(bRecent)) {
          return aRecent - bRecent;
        }
        return Number.isFinite(aRecent) ? -1 : 1;
      }
      const aIsRec = a.quant === effectiveRecommended;
      const bIsRec = b.quant === effectiveRecommended;
      if (aIsRec !== bIsRec) return aIsRec ? -1 : 1;
      return aTier === 0
        ? b.size_bytes - a.size_bytes
        : a.size_bytes - b.size_bytes;
    });
  }, [availableVariants, effectiveRecommended, getGgufFit, recentRank, repoId]);

  if (loading) {
    return (
      <div className="flex items-center gap-2 px-5 py-2">
        <Spinner className="size-3 text-muted-foreground" />
        <span className="text-xs text-muted-foreground">Loading variants…</span>
      </div>
    );
  }

  if (error) {
    return <div className="px-5 py-2 text-xs text-destructive">{error}</div>;
  }

  if (sortedVariants.length === 0) {
    return (
      <div className="px-5 py-2 text-xs text-muted-foreground">
        No on-device quantizations.
      </div>
    );
  }

  return (
    <div className="my-1 ml-3 border-l border-border/40 pl-3">
      <div className="flex items-center gap-1.5 px-3 py-1.5">
        <span className="text-[10px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
          Quantizations
        </span>
        {hasVision && (
          <span className={cn(FORMAT_CHIP_BASE, FORMAT_CHIP_GGUF)}>Vision</span>
        )}
      </div>
      {sortedVariants.map((v) => {
        const fit = getGgufFit(v.size_bytes);
        const oom = fit === "oom";
        const tight = fit === "tight";
        return (
          <button
            key={v.filename}
            type="button"
            onClick={() =>
              handleVariantClick(
                v.quant,
                v.downloaded,
                v.partial,
                v.download_size_bytes ?? v.size_bytes,
              )
            }
            className="picker-row group/qrow flex w-full min-w-0 items-center justify-between gap-2 rounded-[12px] px-3 py-1.5 text-left"
          >
            <span className="min-w-0 flex-1 truncate">
              <span
                className={cn(
                  "font-mono text-[12px]",
                  oom ? "text-muted-foreground/70" : "text-format-gguf",
                )}
              >
                {v.quant}
              </span>
              {v.quant === effectiveRecommended && (
                <span className="ml-1.5 text-[9.5px] font-medium text-primary/70">
                  recommended
                </span>
              )}
            </span>
            <span className="flex shrink-0 items-center gap-1.5">
              {oom && (
                <span className="rounded-[6px] !bg-red-50 px-1.5 py-0.5 text-[9px] font-medium !text-red-700 dark:!bg-red-950 dark:!text-red-400">
                  OOM
                </span>
              )}
              {tight && (
                <span className="text-[9px] font-medium !text-amber-400">
                  TIGHT
                </span>
              )}
              {v.partial && (
                <span className="text-[9px] font-medium text-status-warning">
                  partial
                </span>
              )}
              <span className="text-[10px] text-muted-foreground tabular-nums">
                {formatBytes(v.size_bytes)}
              </span>
              <HugeiconsIcon
                icon={ArrowRight01Icon}
                strokeWidth={1.75}
                className="size-3.5 shrink-0 text-muted-foreground/40 transition-all group-hover/qrow:translate-x-0.5 group-hover/qrow:text-foreground"
              />
            </span>
          </button>
        );
      })}
    </div>
  );
}

// ── Detect GGUF repos by naming convention or hub tag ────────────────────

function isGgufRepo(id: string, hintedIsGguf?: boolean): boolean {
  return Boolean(hintedIsGguf) || hasGgufRepoSuffix(id);
}

function formatToneForModel(
  modelFormat: LocalInventoryRow["modelFormat"],
): FormatDotTone {
  if (modelFormat === "gguf") return "gguf";
  if (modelFormat === "adapter") return "adapter";
  return "checkpoint";
}

// ── Hub Model Picker ──────────────────────────────────────────

export function HubModelPicker({
  value,
  onPick,
  trainedModels = [],
  enabled = true,
}: {
  value?: string;
  onPick: (target: ModelPickTarget) => void;
  /** Studio training & export outputs forwarded from the Train tab data source. */
  trainedModels?: LoraModelOption[];
  enabled?: boolean;
}) {
  const navigate = useNavigate();
  const gpu = useGpuInfo();
  const chatOnly = usePlatformStore((s) => s.isChatOnly());
  const deviceType = usePlatformStore((s) => s.deviceType);

  const [expandedGguf, setExpandedGguf] = useState<string | null>(null);
  const [filter, setFilter] = useState("");

  const {
    cachedGguf,
    cachedModels,
    cachedReady,
    inventoryError,
    localModels,
    refreshInventory,
  } = useChatPickerInventory({ enabled });
  const selectActiveDownloadRefs = useMemo(
    () => createActiveModelDownloadRefsSelector(),
    [],
  );
  const activeDownloadRefs = useDownloadManagerStore(selectActiveDownloadRefs);

  const handleManageOnDevice = useCallback(() => {
    void navigate({
      to: "/models",
      search: { tab: "downloaded" },
    });
  }, [navigate]);

  const filterTokens = useMemo(() => tokenizeQuery(filter), [filter]);
  const matchesFilter = useCallback(
    (id: string) => matchTokens(id, filterTokens),
    [filterTokens],
  );

  const isCachedChatCompatible = useCallback(
    (repo: CachedInventoryRow): boolean =>
      cachedInventoryRowCanChat(repo, deviceType),
    [deviceType],
  );

  const filteredGguf = useMemo(
    () =>
      cachedGguf.filter(
        (c) => matchesFilter(c.repoId) && isCachedChatCompatible(c),
      ),
    [cachedGguf, isCachedChatCompatible, matchesFilter],
  );
  const filteredModels = useMemo(
    () =>
      cachedModels.filter(
        (c) => matchesFilter(c.repoId) && isCachedChatCompatible(c),
      ),
    [cachedModels, isCachedChatCompatible, matchesFilter],
  );
  const filteredLocalModels = useMemo(
    () =>
      localModels.filter((m) => {
        if (m.partial) return false;
        const label = localModelDisplayName(m);
        const searchText = `${label} ${m.title} ${m.repoId ?? ""} ${
          m.loadId
        } ${m.baseModel ?? ""} ${m.path}`;
        if (!matchesFilter(searchText)) return false;
        return localInventoryRowCanChat(m, deviceType);
      }),
    [deviceType, localModels, matchesFilter],
  );

  const recentRank = useRecentRank();
  const sortedGguf = useMemo(
    () =>
      [...filteredGguf].sort((a, b) => {
        const ra = recentRank.rank(a.repoId);
        const rb = recentRank.rank(b.repoId);
        if (ra !== rb) return ra - rb;
        return a.repoId.localeCompare(b.repoId);
      }),
    [filteredGguf, recentRank],
  );
  const sortedModels = useMemo(
    () =>
      [...filteredModels].sort((a, b) => {
        const ra = recentRank.rank(a.repoId);
        const rb = recentRank.rank(b.repoId);
        if (ra !== rb) return ra - rb;
        return a.repoId.localeCompare(b.repoId);
      }),
    [filteredModels, recentRank],
  );
  const sortedLocalModels = useMemo(
    () =>
      [...filteredLocalModels].sort((a, b) => {
        const ra = recentRank.rank(a.id);
        const rb = recentRank.rank(b.id);
        if (ra !== rb) return ra - rb;
        return localModelDisplayName(a).localeCompare(localModelDisplayName(b));
      }),
    [filteredLocalModels, recentRank],
  );

  // Studio outputs that are standalone full models also belong in On Device.
  // Adapters remain Train-only because they need a base model at load time.
  const sortedTrainedOutputs = useMemo(() => {
    const candidates = trainedModels.filter(
      (m) => isStudioTrainedModel(m) && m.exportType !== "lora",
    );
    return candidates
      .filter((m) => {
        const text = `${m.runDisplayName ?? ""} ${m.name} ${m.id} ${
          m.baseModel ?? ""
        }`;
        return matchesFilter(text);
      })
      .sort((a, b) => {
        const ra = recentRank.rank(a.id);
        const rb = recentRank.rank(b.id);
        if (ra !== rb) return ra - rb;
        const at = a.updatedAt ?? -1;
        const bt = b.updatedAt ?? -1;
        if (at !== bt) return bt - at;
        return (a.runDisplayName ?? a.name).localeCompare(
          b.runDisplayName ?? b.name,
        );
      });
  }, [trainedModels, matchesFilter, recentRank]);

  const activeDownloads = useMemo(
    () =>
      activeDownloadRefs.filter((job) =>
        matchesFilter(`${job.repoId} ${job.variant ?? ""}`),
      ),
    [activeDownloadRefs, matchesFilter],
  );

  const hasFilteredDownloaded =
    sortedGguf.length > 0 || sortedModels.length > 0;
  const hasFilteredLocalModels = sortedLocalModels.length > 0;
  const hasFilteredTrainedOutputs = sortedTrainedOutputs.length > 0;
  const hasActiveDownloads = activeDownloads.length > 0;
  const hasAnyInventoryRows =
    cachedGguf.length > 0 ||
    cachedModels.length > 0 ||
    localModels.length > 0 ||
    sortedTrainedOutputs.length > 0 ||
    hasActiveDownloads;
  const localModelGroups = useMemo(
    () =>
      ["hf_cache", "lmstudio", "ollama", "models_dir", "custom"]
        .map((source) => ({
          label: localModelGroupLabel(source as LocalSource),
          models: sortedLocalModels.filter((model) => model.source === source),
        }))
        .filter((group) => group.models.length > 0),
    [sortedLocalModels],
  );

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <div className="relative min-w-0 flex-1">
          <HugeiconsIcon
            icon={Search01Icon}
            strokeWidth={1.8}
            className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground"
          />
          <Input
            value={filter}
            onChange={(event) => setFilter(event.target.value)}
            placeholder="Search on-device models"
            className="field-soft h-9 rounded-full pl-10 pr-3 text-[12.5px] placeholder:text-muted-foreground/80"
          />
        </div>
        {chatOnly ? null : (
          <Tooltip>
            <TooltipTrigger asChild={true}>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={handleManageOnDevice}
                aria-label="Manage on-device models in Hub"
                className="h-9 rounded-full border-border/70 bg-background/70 px-3 text-[12.5px] text-muted-foreground shadow-none hover:bg-muted/70 hover:text-foreground"
              >
                <HugeiconsIcon
                  icon={CubeIcon}
                  strokeWidth={1.75}
                  data-icon="inline-start"
                  className="size-3.5"
                />
                Manage
              </Button>
            </TooltipTrigger>
            <TooltipContent
              side="right"
              sideOffset={8}
              className="tooltip-compact"
            >
              Open Hub On Device
            </TooltipContent>
          </Tooltip>
        )}
      </div>

      <div className="max-h-72 overflow-y-auto">
        <div className="py-1">
          {hasActiveDownloads && (
            <div>
              <ListLabel>Downloading</ListLabel>
              {activeDownloads.map((job) => (
                <DownloadingModelRow key={job.key} jobKey={job.key} />
              ))}
            </div>
          )}

          {cachedReady ? (
            <>
              {inventoryError && (
                <InventoryScanNotice
                  error={inventoryError}
                  hasRows={hasAnyInventoryRows}
                  onRetry={() => void refreshInventory()}
                />
              )}

              {hasFilteredDownloaded && (
                <>
                  {sortedGguf.map((c) => (
                    <div key={c.id}>
                      <ModelRow
                        label={c.repoId}
                        meta={formatBytes(c.bytes)}
                        selected={value === c.loadId}
                        expanded={expandedGguf === c.id}
                        formatTone="gguf"
                        onClick={() =>
                          setExpandedGguf((prev) =>
                            prev === c.id ? null : c.id,
                          )
                        }
                        vramStatus={null}
                      />
                      {expandedGguf === c.id && (
                        <GgufVariantExpander
                          repoId={c.loadId}
                          displayName={c.repoId}
                          onPick={onPick}
                          preferLocalCache={true}
                          localPath={c.cachePath ?? null}
                          gpuGb={gpu.available ? gpu.memoryTotalGb : undefined}
                          systemRamGb={
                            gpu.available ? gpu.systemRamAvailableGb : undefined
                          }
                        />
                      )}
                    </div>
                  ))}
                  {sortedModels.map((c) => (
                    <ModelRow
                      key={c.id}
                      label={c.repoId}
                      meta={formatBytes(c.bytes)}
                      selected={value === c.loadId}
                      formatTone={formatToneForModel(c.modelFormat)}
                      showChevron={false}
                      onClick={() =>
                        onPick({
                          id: c.loadId,
                          displayName: c.repoId,
                          isGguf: false,
                          supportsTrustRemoteCode: true,
                          meta: {
                            source: "hub",
                            isLora: false,
                            isDownloaded: true,
                            preferLocalCache: true,
                            localPath: c.cachePath ?? null,
                            modelFormat: c.modelFormat,
                          },
                        })
                      }
                      vramStatus={null}
                    />
                  ))}
                </>
              )}

              {hasFilteredLocalModels && (
                <>
                  {localModelGroups.map((group) => (
                    <div key={group.label}>
                      <ListLabel>{group.label}</ListLabel>
                      {group.models.map((m) => {
                        const isGguf = m.modelFormat === "gguf";
                        const isGgufDir =
                          isGguf && m.capabilities.requiresVariant;
                        const label = localModelDisplayName(m);
                        return (
                          <div key={m.id}>
                            <ModelRow
                              label={label}
                              meta={localModelGroupLabel(m.source)}
                              selected={value === m.loadId}
                              expanded={isGgufDir && expandedGguf === m.id}
                              formatTone={formatToneForModel(m.modelFormat)}
                              showChevron={isGgufDir}
                              onClick={() => {
                                if (isGgufDir) {
                                  setExpandedGguf((prev) =>
                                    prev === m.id ? null : m.id,
                                  );
                                } else {
                                  onPick({
                                    id: m.loadId,
                                    displayName: label,
                                    isGguf,
                                    supportsTrustRemoteCode: !isGguf,
                                    meta: {
                                      source: "local",
                                      isLora: false,
                                      isDownloaded: true,
                                      localPath: m.path,
                                      modelFormat: m.modelFormat,
                                    },
                                  });
                                }
                              }}
                              vramStatus={null}
                            />
                            {isGgufDir && expandedGguf === m.id && (
                              <GgufVariantExpander
                                repoId={m.loadId}
                                displayName={label}
                                onPick={onPick}
                                sourceOverride="local"
                                localPath={m.path}
                                gpuGb={
                                  gpu.available ? gpu.memoryTotalGb : undefined
                                }
                                systemRamGb={
                                  gpu.available
                                    ? gpu.systemRamAvailableGb
                                    : undefined
                                }
                              />
                            )}
                          </div>
                        );
                      })}
                    </div>
                  ))}
                </>
              )}

              {hasFilteredTrainedOutputs && (
                <div>
                  <ListLabel>Trained outputs</ListLabel>
                  {sortedTrainedOutputs.map((m) => {
                    const isGguf = m.exportType === "gguf";
                    const label = m.runDisplayName?.trim() || m.name;
                    const sourceLabel = trainedOutputMeta(m);
                    // Map LoraModelOption.source onto the picker meta vocab so the
                    // downstream delete flow keeps working: "training" outputs are
                    // routed through meta.source="lora" (the existing convention
                    // for studio training-runs in pickerSourceToDeleteSource) and
                    // "exported" stays "exported".
                    const pickSource: ModelSelectorChangeMeta["source"] =
                      m.source === "exported" ? "exported" : "lora";
                    return (
                      <div key={m.id}>
                        <ModelRow
                          label={label}
                          meta={sourceLabel}
                          selected={value === m.id}
                          expanded={isGguf && expandedGguf === m.id}
                          formatTone={isGguf ? "gguf" : "checkpoint"}
                          showChevron={isGguf}
                          tooltipText={
                            m.baseModel ? `Base: ${m.baseModel}` : undefined
                          }
                          onClick={() => {
                            if (isGguf) {
                              setExpandedGguf((prev) =>
                                prev === m.id ? null : m.id,
                              );
                              return;
                            }
                            onPick({
                              id: m.id,
                              displayName: label,
                              isGguf,
                              supportsTrustRemoteCode: !isGguf,
                              meta: {
                                source: pickSource,
                                isLora: false,
                                exportType: m.exportType,
                                isDownloaded: true,
                                localPath: m.id,
                                modelFormat: "safetensors",
                              },
                            });
                          }}
                          vramStatus={null}
                        />
                        {isGguf && expandedGguf === m.id && (
                          <GgufVariantExpander
                            repoId={m.id}
                            displayName={label}
                            onPick={onPick}
                            sourceOverride={pickSource}
                            exportType="gguf"
                            gpuGb={
                              gpu.available ? gpu.memoryTotalGb : undefined
                            }
                            systemRamGb={
                              gpu.available
                                ? gpu.systemRamAvailableGb
                                : undefined
                            }
                          />
                        )}
                      </div>
                    );
                  })}
                </div>
              )}

              {!hasActiveDownloads &&
                !hasFilteredDownloaded &&
                !hasFilteredLocalModels &&
                !hasFilteredTrainedOutputs && (
                  <div className="px-2.5 py-3 text-xs text-muted-foreground">
                    {filter.trim().length > 0
                      ? "No matching models."
                      : inventoryError
                        ? "No models could be shown."
                        : "No on-device models yet."}
                  </div>
                )}
            </>
          ) : (
            <div className="flex items-center gap-2 px-5 py-3">
              <Spinner className="size-3 text-muted-foreground" />
              <span className="text-xs text-muted-foreground">
                Loading models…
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function describeAdapter(adapter: LoraModelOption) {
  const isLocal = adapter.source === "local";
  const isTraining = adapter.source === "training";
  const isExported = adapter.source === "exported";
  const isStudioTrained = isStudioTrainedModel(adapter);
  const isMerged = adapter.exportType === "merged";
  const isGguf = adapter.exportType === "gguf";
  const isExportedGguf = isExported && isGguf;
  const isTrainingFull = isTraining && isMerged;
  const isLocalGgufDir =
    isLocal && (isGgufRepo(adapter.id) || isGgufRepo(adapter.name));
  const trainedMeta = isStudioTrained ? trainedOutputMeta(adapter) : null;
  const tag = isLocal
    ? isLocalGgufDir
      ? "GGUF"
      : "Local"
    : isGguf
      ? "GGUF"
      : isMerged
        ? "Safetensors"
        : trainedMeta
          ? (trainingMethodLabel(adapter.trainingMethod) ?? "Adapter")
          : "LoRA";
  const meta = isLocal
    ? isLocalGgufDir
      ? "GGUF"
      : "Local"
    : isTrainingFull
      ? (trainedMeta ?? "Full · Safetensors")
      : isExported
        ? (trainedMeta ?? `${tag} · Exported`)
        : (trainedMeta ?? tag);
  return {
    isLocal,
    isExported,
    isTraining,
    isMerged,
    isGguf,
    isExportedGguf,
    isLocalGgufDir,
    tag,
    meta,
    isExpandableGguf: isLocalGgufDir || isExportedGguf,
  };
}

function adapterTitle(adapter: LoraModelOption): string {
  const run = adapter.runDisplayName?.trim();
  if (run && run.length > 0) return run;
  return adapter.name;
}

export function LoraModelPicker({
  loraModels,
  value,
  onPick,
}: {
  loraModels: LoraModelOption[];
  value?: string;
  onPick: (target: ModelPickTarget) => void;
}) {
  const [query, setQuery] = useState("");
  const [expandedBase, setExpandedBase] = useState<string | null>(null);
  const [expandedGguf, setExpandedGguf] = useState<string | null>(null);
  const gpu = useGpuInfo();

  const recentRank = useRecentRank();

  // Train is the user's training-output history. It includes adapters and
  // standalone trained checkpoints, while arbitrary local/base models stay out.
  const trainedOutputs = useMemo(
    () => loraModels.filter((m) => isStudioTrainedModel(m)),
    [loraModels],
  );

  const normalized = useMemo(
    () =>
      trainedOutputs
        .map((model) => ({
          ...model,
          baseModel:
            model.baseModel || model.description || "Unknown base model",
        }))
        .sort((a, b) => {
          const baseCmp = a.baseModel.localeCompare(b.baseModel);
          if (baseCmp !== 0) return baseCmp;
          if (a.baseModel === "External" && b.baseModel === "External") {
            const aUnsloth = a.name.startsWith("unsloth/") ? 0 : 1;
            const bUnsloth = b.name.startsWith("unsloth/") ? 0 : 1;
            if (aUnsloth !== bUnsloth) return aUnsloth - bUnsloth;
          }
          const aTime = a.updatedAt ?? -1;
          const bTime = b.updatedAt ?? -1;
          if (aTime !== bTime) return bTime - aTime;
          return a.name.localeCompare(b.name);
        }),
    [trainedOutputs],
  );

  const queryTokens = useMemo(() => tokenizeQuery(query), [query]);
  const isSearching = queryTokens.length > 0;
  const grouped = useMemo(() => {
    const out = new Map<string, LoraModelOption[]>();

    for (const model of normalized) {
      const searchText = `${model.name} ${model.baseModel} ${model.id} ${
        model.runDisplayName ?? ""
      }`;
      if (!matchTokens(searchText, queryTokens)) continue;

      const key = model.baseModel || "Unknown base model";
      const prev = out.get(key) ?? [];
      prev.push(model);
      out.set(key, prev);
    }

    for (const adapters of out.values()) {
      adapters.sort((a, b) => {
        const ra = recentRank.rank(a.id);
        const rb = recentRank.rank(b.id);
        if (ra !== rb) return ra - rb;
        const aTime = a.updatedAt ?? -1;
        const bTime = b.updatedAt ?? -1;
        if (aTime !== bTime) return bTime - aTime;
        return a.name.localeCompare(b.name);
      });
    }

    return [...out.entries()].sort((a, b) => {
      const aRank = Math.min(...a[1].map((model) => recentRank.rank(model.id)));
      const bRank = Math.min(...b[1].map((model) => recentRank.rank(model.id)));
      if (aRank !== bRank) return aRank - bRank;
      const aLatest = Math.max(...a[1].map((model) => model.updatedAt ?? -1));
      const bLatest = Math.max(...b[1].map((model) => model.updatedAt ?? -1));
      if (aLatest !== bLatest) return bLatest - aLatest;
      return a[0].localeCompare(b[0]);
    });
  }, [normalized, queryTokens, recentRank]);

  return (
    <div className="space-y-2">
      <div className="relative">
        <HugeiconsIcon
          icon={Search01Icon}
          strokeWidth={1.8}
          className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground"
        />
        <Input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search trained models"
          className="field-soft h-9 rounded-full pl-10 pr-3 text-[12.5px] placeholder:text-muted-foreground/80"
        />
      </div>

      <div className="max-h-72 overflow-y-auto">
        <div className="py-1">
          {grouped.length === 0 ? (
            <div className="px-2.5 py-2 text-xs text-muted-foreground">
              No trained models found.
            </div>
          ) : (
            grouped.map(([baseModel, adapters]) => {
              const isOpen = isSearching || expandedBase === baseModel;
              const adapterCount = adapters.length;
              const countLabel = `${adapterCount} ${
                adapterCount === 1 ? "run" : "runs"
              }`;
              const containsSelected = adapters.some((a) => a.id === value);
              return (
                <div key={baseModel}>
                  <ModelRow
                    label={baseModel}
                    meta={countLabel}
                    selected={containsSelected && !isOpen}
                    expanded={isOpen}
                    onClick={() => {
                      if (isSearching) return;
                      setExpandedBase((prev) =>
                        prev === baseModel ? null : baseModel,
                      );
                    }}
                    vramStatus={null}
                  />
                  {isOpen && (
                    <div className="my-1 ml-3 border-l border-border/40 pl-3">
                      {adapters.map((adapter) => {
                        const info = describeAdapter(adapter);
                        const title = adapterTitle(adapter);
                        const hasRunName =
                          !!adapter.runDisplayName &&
                          adapter.runDisplayName !== adapter.name;
                        const ggufExpanded =
                          info.isExpandableGguf && expandedGguf === adapter.id;
                        return (
                          <div key={adapter.id}>
                            <Tooltip>
                              <TooltipTrigger asChild={true}>
                                <button
                                  type="button"
                                  data-selected={
                                    value === adapter.id || undefined
                                  }
                                  aria-expanded={ggufExpanded || undefined}
                                  onClick={() => {
                                    if (info.isExpandableGguf) {
                                      setExpandedGguf((prev) =>
                                        prev === adapter.id ? null : adapter.id,
                                      );
                                      return;
                                    }
                                    const source = info.isLocal
                                      ? "local"
                                      : info.isExported
                                        ? "exported"
                                        : "lora";
                                    onPick({
                                      id: adapter.id,
                                      displayName: title,
                                      isGguf: info.isGguf,
                                      supportsTrustRemoteCode:
                                        !info.isLocal && !info.isGguf,
                                      meta: {
                                        source,
                                        isLora:
                                          !info.isLocal &&
                                          !info.isMerged &&
                                          !info.isGguf,
                                        exportType: adapter.exportType,
                                        isDownloaded: true,
                                        modelFormat: info.isGguf
                                          ? "gguf"
                                          : info.isMerged
                                            ? "safetensors"
                                            : "adapter",
                                      },
                                    });
                                  }}
                                  className="picker-row group/arow flex w-full min-w-0 items-center gap-2 rounded-[12px] px-3 py-1.5 text-left"
                                >
                                  <span className="flex min-w-0 flex-1 flex-col leading-tight">
                                    <span className="truncate text-[12.5px] font-medium text-foreground">
                                      {title}
                                    </span>
                                    {hasRunName && (
                                      <span className="truncate text-[10.5px] text-muted-foreground/85">
                                        {adapter.name}
                                      </span>
                                    )}
                                  </span>
                                  <span className="ml-auto flex shrink-0 items-center gap-1.5">
                                    <FormatChip meta={info.meta} />
                                    <HugeiconsIcon
                                      icon={ArrowRight01Icon}
                                      strokeWidth={1.75}
                                      className={cn(
                                        "size-3.5 shrink-0 text-muted-foreground/40 transition-all",
                                        ggufExpanded
                                          ? "rotate-90 text-foreground"
                                          : "group-hover/arow:translate-x-0.5 group-hover/arow:text-foreground",
                                      )}
                                    />
                                  </span>
                                </button>
                              </TooltipTrigger>
                              <TooltipContent
                                side="right"
                                align="start"
                                sideOffset={8}
                                className="tooltip-compact max-w-xs break-all"
                              >
                                <span className="block break-words">
                                  {title}
                                </span>
                                {hasRunName && (
                                  <span className="mt-1 block break-words text-[10px]">
                                    Model: {adapter.name}
                                  </span>
                                )}
                                <span className="mt-1 block break-all text-[10px] text-muted-foreground">
                                  {adapter.id}
                                </span>
                              </TooltipContent>
                            </Tooltip>
                            {ggufExpanded && (
                              <GgufVariantExpander
                                repoId={adapter.id}
                                displayName={title}
                                onPick={onPick}
                                gpuGb={
                                  gpu.available ? gpu.memoryTotalGb : undefined
                                }
                                systemRamGb={
                                  gpu.available
                                    ? gpu.systemRamAvailableGb
                                    : undefined
                                }
                                sourceOverride={
                                  info.isExportedGguf ? "exported" : undefined
                                }
                                exportType={
                                  info.isExportedGguf ? "gguf" : undefined
                                }
                              />
                            )}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
}
