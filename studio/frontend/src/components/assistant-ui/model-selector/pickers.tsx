// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Input } from "@/components/ui/input";
import { matchTokens, tokenizeQuery } from "@/lib/search-text";
import { Spinner } from "@/components/ui/spinner";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { usePlatformStore } from "@/config/env";
import {
  listCachedGguf,
  listCachedModels,
  listGgufVariants,
  listLocalModels,
} from "@/features/chat/api/chat-api";
import type {
  CachedGgufRepo,
  CachedModelRepo,
  LocalModelInfo,
} from "@/features/chat/api/chat-api";
import type { GgufVariantDetail } from "@/features/chat/types/api";
import { formatBytes, useInventoryVersion } from "@/features/models";
import { classifyUnslothSupport, useGpuInfo } from "@/hooks";
import { cn } from "@/lib/utils";
import { classifyGgufFit, ggufFitTier, type GgufFitTier } from "@/lib/gguf-fit";
import type { VramFitStatus } from "@/lib/vram";
import { useHfTokenStore } from "@/stores/hf-token-store";
import {
  ArrowRight01Icon,
  Search01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type {
  LoraModelOption,
  ModelPickTarget,
  ModelSelectorChangeMeta,
} from "./types";
import { buildRecentRank } from "@/features/chat/model-config/recent-models";

function ListLabel({ children }: { children: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-1 px-3 pt-2 pb-1">
      <span className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
        {children}
      </span>
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
  return <span className={cn(FORMAT_CHIP_BASE, FORMAT_CHIP_DEFAULT)}>{meta}</span>;
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
      <ModelLabel
        label={label}
        className={cn(
          "block min-w-0 flex-1",
          exceeds && "opacity-60",
        )}
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
              expanded
                ? "rotate-90"
                : "group-hover/row:translate-x-0.5",
            )}
          />
        )}
      </span>
    </button>
  );

  if (vramTooltipText) {
    return (
      <Tooltip>
        <TooltipTrigger asChild={true}>{content}</TooltipTrigger>
        <TooltipContent
          side="left"
          className="tooltip-compact max-w-xs break-all"
        >
          {label}
          <span className="block text-[10px] mt-1">{vramTooltipText}</span>
        </TooltipContent>
      </Tooltip>
    );
  }

  if (tooltipText) {
    return (
      <Tooltip>
        <TooltipTrigger asChild={true}>{content}</TooltipTrigger>
        <TooltipContent
          side="left"
          className="tooltip-compact max-w-xs break-all"
        >
          {tooltipText}
        </TooltipContent>
      </Tooltip>
    );
  }
  return content;
}

// ── GGUF Variant Expander ────────────────────────────────────

function GgufVariantExpander({
  repoId,
  displayName,
  onPick,
  gpuGb,
  systemRamGb,
  sourceOverride,
}: {
  repoId: string;
  displayName: string;
  onPick: (target: ModelPickTarget) => void;
  gpuGb?: number;
  systemRamGb?: number;
  sourceOverride?: ModelSelectorChangeMeta["source"];
}) {
  const [variants, setVariants] = useState<GgufVariantDetail[] | null>(null);
  const [defaultVariant, setDefaultVariant] = useState<string | null>(null);
  const [hasVision, setHasVision] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const hfToken = useHfTokenStore((s) => s.token) || undefined;

  useEffect(() => {
    let canceled = false;
    setLoading(true);
    setError(null);

    listGgufVariants(repoId, hfToken)
      .then((res) => {
        if (canceled) return;
        setVariants(res.variants);
        setDefaultVariant(res.default_variant);
        setHasVision(res.has_vision);
      })
      .catch((err) => {
        if (canceled) return;
        setError(
          err instanceof Error ? err.message : "Failed to load variants",
        );
      })
      .finally(() => {
        if (!canceled) setLoading(false);
      });

    return () => {
      canceled = true;
    };
  }, [repoId, hfToken]);

  // Covers Unix absolute (/), Windows drive (C:\, D:/), UNC (\\server), relative (./, ../), tilde (~/)
  const isLocalPath = /^(\/|\.{1,2}[\\\/]|~[\\\/]|[A-Za-z]:[\\\/]|\\\\)/.test(
    repoId,
  );

  const handleVariantClick = useCallback(
    (quant: string, downloaded?: boolean, sizeBytes?: number) => {
      onPick({
        id: repoId,
        displayName,
        isGguf: true,
        supportsTrustRemoteCode: false,
        meta: {
          source: sourceOverride ?? (isLocalPath ? "local" : "hub"),
          isLora: false,
          ggufVariant: quant,
          isDownloaded: isLocalPath ? true : downloaded,
          expectedBytes: sizeBytes,
        },
      });
    },
    [displayName, isLocalPath, onPick, repoId, sourceOverride],
  );

  const getGgufFit = useCallback(
    (sizeBytes: number): GgufFitTier =>
      ggufFitTier(classifyGgufFit(sizeBytes, { gpuGb, systemRamGb })),
    [gpuGb, systemRamGb],
  );

  const downloadedVariants = useMemo(
    () => (variants ?? []).filter((v) => v.downloaded),
    [variants],
  );

  const effectiveRecommended = useMemo(() => {
    if (downloadedVariants.length === 0) return null;
    if (!gpuGb || gpuGb <= 0) return defaultVariant;
    const defaultV = downloadedVariants.find((v) => v.quant === defaultVariant);
    if (defaultV && getGgufFit(defaultV.size_bytes) !== "oom") return defaultVariant;
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
    return [...downloadedVariants].sort((a, b) => {
      const aTier = tierOf(a);
      const bTier = tierOf(b);
      if (aTier !== bTier) return aTier - bTier;
      const aIsRec = a.quant === effectiveRecommended;
      const bIsRec = b.quant === effectiveRecommended;
      if (aIsRec !== bIsRec) return aIsRec ? -1 : 1;
      return aTier === 0
        ? b.size_bytes - a.size_bytes
        : a.size_bytes - b.size_bytes;
    });
  }, [downloadedVariants, effectiveRecommended, getGgufFit]);

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
              handleVariantClick(v.quant, v.downloaded, v.size_bytes)
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

function hasGgufSuffix(id: string): boolean {
  return /-GGUF(?:$|-)/i.test(id);
}

function isGgufRepo(id: string, hintedIsGguf?: boolean): boolean {
  return Boolean(hintedIsGguf) || hasGgufSuffix(id);
}

// Module-level caches so re-mounting the popover shows results instantly
let _cachedGgufCache: CachedGgufRepo[] = [];
let _cachedModelsCache: CachedModelRepo[] = [];
let _lmStudioCache: LocalModelInfo[] = [];
let _cachedInventoryVersion = -1;
// The cached-repo lists are HF-token-scoped (gated repos differ per token), so
// the cache is only fresh when both the inventory version AND the token match.
let _cachedToken: string | null = null;

/** Sort external (lmstudio-source) models with unsloth publisher first. */
function sortLmStudio(models: LocalModelInfo[]): LocalModelInfo[] {
  return [...models].sort((a, b) => {
    const aUnsloth = (a.model_id ?? "").startsWith("unsloth/") ? 0 : 1;
    const bUnsloth = (b.model_id ?? "").startsWith("unsloth/") ? 0 : 1;
    if (aUnsloth !== bUnsloth) return aUnsloth - bUnsloth;
    return (a.model_id ?? a.display_name).localeCompare(
      b.model_id ?? b.display_name,
    );
  });
}

// ── Hub Model Picker ──────────────────────────────────────────

export function HubModelPicker({
  value,
  onPick,
}: {
  value?: string;
  onPick: (target: ModelPickTarget) => void;
}) {
  const gpu = useGpuInfo();
  const chatOnly = usePlatformStore((s) => s.isChatOnly());
  const deviceType = usePlatformStore((s) => s.deviceType);

  const [expandedGguf, setExpandedGguf] = useState<string | null>(null);
  const [filter, setFilter] = useState("");

  const hfToken = useHfTokenStore((s) => s.token) || undefined;
  const inventoryVersion = useInventoryVersion();
  const cacheFresh =
    _cachedInventoryVersion === inventoryVersion &&
    _cachedToken === (hfToken ?? null);

  const [cachedGguf, setCachedGguf] =
    useState<CachedGgufRepo[]>(_cachedGgufCache);
  const [cachedModels, setCachedModels] =
    useState<CachedModelRepo[]>(_cachedModelsCache);
  const alreadyCached =
    cacheFresh &&
    (_cachedGgufCache.length > 0 || _cachedModelsCache.length > 0);
  const [cachedReady, setCachedReady] = useState(alreadyCached);
  const [lmStudioModels, setLmStudioModels] =
    useState<LocalModelInfo[]>(_lmStudioCache);

  const refreshLocalModelsList = useCallback(() => {
    listLocalModels()
      .then((res) => {
        const lm = sortLmStudio(
          res.models.filter((m) => m.source === "lmstudio"),
        );
        _lmStudioCache = lm;
        setLmStudioModels(lm);
      })
      .catch(() => {});
  }, []);

  const loadedKeyRef = useRef<string | null>(null);

  useEffect(() => {
    const loadKey = `${inventoryVersion}::${hfToken ?? ""}`;
    if (loadedKeyRef.current === loadKey) return;
    loadedKeyRef.current = loadKey;
    const controller = new AbortController();
    const { signal } = controller;
    refreshLocalModelsList();
    let done = 0;
    const check = () => {
      if (signal.aborted) return;
      if (++done >= 2) {
        setCachedReady(true);
        _cachedInventoryVersion = inventoryVersion;
        _cachedToken = hfToken ?? null;
      }
    };
    listCachedGguf(hfToken, signal)
      .then((v) => {
        if (signal.aborted) return;
        _cachedGgufCache = v;
        setCachedGguf(v);
      })
      .catch(() => {})
      .finally(check);
    listCachedModels(hfToken, signal)
      .then((v) => {
        if (signal.aborted) return;
        _cachedModelsCache = v;
        setCachedModels(v);
      })
      .catch(() => {})
      .finally(check);
    return () => controller.abort();
  }, [refreshLocalModelsList, inventoryVersion, hfToken]);

  const filterTokens = useMemo(() => tokenizeQuery(filter), [filter]);
  const matchesFilter = useCallback(
    (id: string) => matchTokens(id, filterTokens),
    [filterTokens],
  );

  const isRepoCompatible = useCallback(
    (
      repo: {
        repo_id: string;
        pipeline_tag?: string | null;
        tags?: string[];
        library_name?: string | null;
      },
    ): boolean => {
      return (
        classifyUnslothSupport({
          modelId: repo.repo_id,
          pipelineTag: repo.pipeline_tag,
          tags: repo.tags,
          libraryName: repo.library_name,
          deviceType,
        }).status !== "unsupported"
      );
    },
    [deviceType],
  );

  const isLocalCompatible = useCallback(
    (...candidates: ReadonlyArray<string | null | undefined>): boolean => {
      for (const candidate of candidates) {
        if (!candidate) continue;
        if (
          classifyUnslothSupport({ modelId: candidate, deviceType }).status ===
          "unsupported"
        ) {
          return false;
        }
      }
      return true;
    },
    [deviceType],
  );

  const filteredGguf = useMemo(
    () =>
      cachedGguf.filter(
        (c) => matchesFilter(c.repo_id) && isRepoCompatible(c),
      ),
    [cachedGguf, isRepoCompatible, matchesFilter],
  );
  const filteredModels = useMemo(
    () =>
      cachedModels.filter(
        (c) => !c.partial && matchesFilter(c.repo_id) && isRepoCompatible(c),
      ),
    [cachedModels, isRepoCompatible, matchesFilter],
  );
  const filteredLmStudio = useMemo(
    () =>
      lmStudioModels.filter((m) => {
        const label = m.model_id ?? m.display_name;
        if (!matchesFilter(label)) return false;
        const pathBase = m.path.split(/[/\\]/).pop() ?? "";
        const isGgufEntry =
          isGgufRepo(m.id) ||
          isGgufRepo(m.display_name) ||
          m.path.toLowerCase().endsWith(".gguf");
        if (isGgufEntry) return true;
        return isLocalCompatible(m.model_id, m.display_name, pathBase);
      }),
    [isLocalCompatible, lmStudioModels, matchesFilter],
  );

  const recentRank = useMemo(() => buildRecentRank(), []);
  const sortedGguf = useMemo(
    () =>
      [...filteredGguf].sort((a, b) => {
        const ra = recentRank.rank(a.repo_id);
        const rb = recentRank.rank(b.repo_id);
        if (ra !== rb) return ra - rb;
        return a.repo_id.localeCompare(b.repo_id);
      }),
    [filteredGguf, recentRank],
  );
  const sortedModels = useMemo(
    () =>
      [...filteredModels].sort((a, b) => {
        const ra = recentRank.rank(a.repo_id);
        const rb = recentRank.rank(b.repo_id);
        if (ra !== rb) return ra - rb;
        return a.repo_id.localeCompare(b.repo_id);
      }),
    [filteredModels, recentRank],
  );
  const sortedLmStudio = useMemo(
    () =>
      [...filteredLmStudio].sort((a, b) => {
        const ra = recentRank.rank(a.id);
        const rb = recentRank.rank(b.id);
        if (ra !== rb) return ra - rb;
        const aLabel = a.model_id ?? a.display_name;
        const bLabel = b.model_id ?? b.display_name;
        return aLabel.localeCompare(bLabel);
      }),
    [filteredLmStudio, recentRank],
  );

  const hasFilteredDownloaded =
    sortedGguf.length > 0 || (!chatOnly && sortedModels.length > 0);
  const hasFilteredLmStudio = chatOnly && sortedLmStudio.length > 0;

  return (
    <div className="space-y-2">
      <div className="relative">
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

      <div className="max-h-72 overflow-y-auto">
        <div className="py-1">
          {!cachedReady ? (
            <div className="flex items-center gap-2 px-5 py-3">
              <Spinner className="size-3 text-muted-foreground" />
              <span className="text-xs text-muted-foreground">
                Loading models…
              </span>
            </div>
          ) : (
            <>
              {hasFilteredDownloaded && (
                <>
                  {sortedGguf.map((c) => (
                    <div key={c.repo_id}>
                      <ModelRow
                        label={c.repo_id}
                        meta={`GGUF · ${formatBytes(c.size_bytes)}`}
                        selected={value === c.repo_id}
                        expanded={expandedGguf === c.repo_id}
                        onClick={() =>
                          setExpandedGguf((prev) =>
                            prev === c.repo_id ? null : c.repo_id,
                          )
                        }
                        vramStatus={null}
                      />
                      {expandedGguf === c.repo_id && (
                        <GgufVariantExpander
                          repoId={c.repo_id}
                          displayName={c.repo_id}
                          onPick={onPick}
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
                  ))}
                  {!chatOnly &&
                    sortedModels.map((c) => (
                      <ModelRow
                        key={c.repo_id}
                        label={c.repo_id}
                        meta={formatBytes(c.size_bytes)}
                        selected={value === c.repo_id}
                        onClick={() =>
                          onPick({
                            id: c.repo_id,
                            displayName: c.repo_id,
                            isGguf: false,
                            supportsTrustRemoteCode: true,
                            meta: {
                              source: "hub",
                              isLora: false,
                              isDownloaded: true,
                            },
                          })
                        }
                        vramStatus={null}
                      />
                    ))}
                </>
              )}

              {hasFilteredLmStudio && (
                <>
                  <ListLabel>External</ListLabel>
                  {sortedLmStudio.map((m) => {
                      // A standalone .gguf file is one weight to load directly;
                      // only a GGUF repo/directory (named *-GGUF) holds multiple
                      // variants to expand. Detecting both keeps the label and
                      // the click path in agreement.
                      const isGgufFile = m.path.toLowerCase().endsWith(".gguf");
                      const isGgufDir =
                        !isGgufFile &&
                        (isGgufRepo(m.id) || isGgufRepo(m.display_name));
                      const isGguf = isGgufFile || isGgufDir;
                      return (
                        <div key={m.id}>
                          <ModelRow
                            label={m.model_id ?? m.display_name}
                            meta={isGguf ? "GGUF" : "Local"}
                            selected={value === m.id}
                            expanded={isGgufDir && expandedGguf === m.id}
                            onClick={() => {
                              if (isGgufDir) {
                                setExpandedGguf((prev) =>
                                  prev === m.id ? null : m.id,
                                );
                              } else {
                                onPick({
                                  id: m.id,
                                  displayName: m.model_id ?? m.display_name,
                                  isGguf: isGgufFile,
                                  supportsTrustRemoteCode: !isGgufFile,
                                  meta: {
                                    source: "local",
                                    isLora: false,
                                    isDownloaded: true,
                                  },
                                });
                              }
                            }}
                            vramStatus={null}
                          />
                          {isGgufDir && expandedGguf === m.id && (
                            <GgufVariantExpander
                              repoId={m.id}
                              displayName={m.model_id ?? m.display_name}
                              onPick={onPick}
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
                </>
              )}

              {!hasFilteredDownloaded && !hasFilteredLmStudio && (
                <div className="px-2.5 py-3 text-xs text-muted-foreground">
                  {filter.trim().length > 0
                    ? "No matching models."
                    : chatOnly
                      ? "No external models found."
                      : "No on-device models yet."}
                </div>
              )}
            </>
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
  const isMerged = adapter.exportType === "merged";
  const isGguf = adapter.exportType === "gguf";
  const isExportedGguf = isExported && isGguf;
  const isTrainingFull = isTraining && isMerged;
  const isLocalGgufDir =
    isLocal && (isGgufRepo(adapter.id) || isGgufRepo(adapter.name));
  const tag = isLocal
    ? isLocalGgufDir
      ? "GGUF"
      : "Local"
    : isGguf
      ? "GGUF"
      : isTrainingFull
        ? "Full"
        : isExported
          ? isMerged
            ? "Merged"
            : "LoRA"
          : "LoRA";
  const meta = isLocal
    ? isLocalGgufDir
      ? "GGUF"
      : "Local"
    : isTrainingFull
      ? "Full finetune"
      : isExported
        ? `${tag} · Exported`
        : tag;
  return {
    isLocal,
    isExported,
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

  const recentRank = useMemo(() => buildRecentRank(), []);

  const normalized = useMemo(
    () =>
      loraModels
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
    [loraModels],
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
      const aRank = Math.min(
        ...a[1].map((model) => recentRank.rank(model.id)),
      );
      const bRank = Math.min(
        ...b[1].map((model) => recentRank.rank(model.id)),
      );
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
          placeholder="Search fine-tuned models"
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
                          info.isExpandableGguf &&
                          expandedGguf === adapter.id;
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
                                        prev === adapter.id
                                          ? null
                                          : adapter.id,
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
                                      displayName: adapter.name,
                                      isGguf: false,
                                      supportsTrustRemoteCode:
                                        !info.isLocal && !info.isGguf,
                                      meta: {
                                        source,
                                        isLora:
                                          !info.isLocal &&
                                          !info.isMerged &&
                                          !info.isGguf,
                                        isDownloaded: true,
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
                                side="left"
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
                                displayName={adapter.name}
                                onPick={onPick}
                                gpuGb={
                                  gpu.available
                                    ? gpu.memoryTotalGb
                                    : undefined
                                }
                                systemRamGb={
                                  gpu.available
                                    ? gpu.systemRamAvailableGb
                                    : undefined
                                }
                                sourceOverride={
                                  info.isExportedGguf ? "exported" : undefined
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
