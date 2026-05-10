// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Spinner } from "@/components/ui/spinner";
import {
  deleteCachedModel,
  getDownloadProgress,
  getGgufDownloadProgress,
  getModelDownloadStatus,
  listGgufVariants,
  startModelDownload,
} from "@/features/chat/api/chat-api";
import type { GgufVariantDetail } from "@/features/chat/types/api";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import {
  ArrowDown01Icon,
  CheckmarkCircle02Icon,
  Delete02Icon,
  Download01Icon,
  InformationCircleIcon,
  PencilEdit02Icon,
  PlayIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import { fetchModelSize } from "../lib/dataset-size";
import { formatBytes } from "../lib/format";

type FitClass = "fits" | "marginal" | "partial" | "oom";

const VRAM_HEADROOM_RATIO = 0.92;
const CONTEXT_OVERHEAD_GB_4K = 1.0;
const ACTIVATIONS_RATIO = 0.15;
const RAM_OFFLOAD_USABLE_RATIO = 0.5;

function requiredVramGb(sizeBytes: number, contextOverheadGb = CONTEXT_OVERHEAD_GB_4K): number {
  const sizeGb = sizeBytes / 1024 ** 3;
  return sizeGb * (1 + ACTIVATIONS_RATIO) + contextOverheadGb;
}

function getFitClass(
  sizeBytes: number,
  gpuGb?: number,
  systemRamGb?: number,
): FitClass {
  if (!gpuGb || gpuGb <= 0) return "fits";
  const required = requiredVramGb(sizeBytes);
  const budget = gpuGb * VRAM_HEADROOM_RATIO;
  if (required <= budget) return "fits";
  if (required <= gpuGb) return "marginal";
  const combined = gpuGb + (systemRamGb ?? 0) * RAM_OFFLOAD_USABLE_RATIO;
  if (required <= combined) return "partial";
  return "oom";
}

function pickRecommendedQuant(
  variants: GgufVariantDetail[],
  gpuGb?: number,
  systemRamGb?: number,
): string | null {
  if (variants.length === 0) return null;
  if (!gpuGb || gpuGb <= 0) return null;
  const bySizeDesc = [...variants].sort((a, b) => b.size_bytes - a.size_bytes);
  for (const cls of ["fits", "marginal", "partial"] as const) {
    const match = bySizeDesc.find(
      (v) => getFitClass(v.size_bytes, gpuGb, systemRamGb) === cls,
    );
    if (match) return match.quant;
  }
  const bySizeAsc = [...variants].sort((a, b) => a.size_bytes - b.size_bytes);
  return bySizeAsc[0]?.quant ?? null;
}

function DownloadProgressBar({
  progress,
}: {
  progress: { expectedBytes: number; downloadedBytes: number; fraction: number };
}) {
  const percent = Math.round(Math.min(progress.fraction, 1) * 100);
  const totalLabel =
    progress.expectedBytes > 0 ? formatBytes(progress.expectedBytes) : null;
  return (
    <div className="flex flex-col gap-1.5">
      <div className="h-1 overflow-hidden rounded-full bg-border/40">
        <div
          className="h-full rounded-full bg-primary transition-[width] duration-300"
          style={{ width: `${percent}%` }}
        />
      </div>
      <div className="flex items-center justify-between text-[10.5px] text-muted-foreground tabular-nums">
        <span>
          {formatBytes(progress.downloadedBytes)}
          {totalLabel && ` / ${totalLabel}`}
        </span>
        <span>{percent}%</span>
      </div>
    </div>
  );
}

interface FitBadgeMeta {
  label: string;
  tooltip: string;
  iconClassName: string;
}

const FIT_BADGE: Record<FitClass, FitBadgeMeta> = {
  fits: {
    label: "Full GPU offload",
    tooltip: "Full offload likely possible on your system.",
    iconClassName: "text-emerald-600 dark:text-emerald-400",
  },
  marginal: {
    label: "Might fit",
    tooltip:
      "Might fit. Within the last GB of VRAM headroom, so loading can fail if other apps are using GPU memory.",
    iconClassName: "text-amber-600 dark:text-amber-400",
  },
  partial: {
    label: "Partial offload",
    tooltip:
      "Partial offload possible. Exceeds VRAM but fits with system RAM offload. Inference will be slower.",
    iconClassName: "text-sky-600 dark:text-sky-400",
  },
  oom: {
    label: "Won't fit",
    tooltip: "Exceeds combined VRAM and system RAM budget.",
    iconClassName: "text-rose-600 dark:text-rose-400",
  },
};

/** Chip styling matching the on-device list's StatChip — no icon. */
const CHIP_BASE =
  "inline-flex h-5 shrink-0 items-center justify-center whitespace-nowrap rounded-[7px] border px-1.5 text-[11.5px] font-medium tabular-nums leading-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]";
const CHIP_DEFAULT =
  "border-foreground/15 bg-muted text-foreground/85 dark:border-border/60 dark:bg-white/[0.04] dark:text-foreground/85";
const CHIP_ACTIVE =
  "border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300";
const CHIP_GGUF =
  "border-blue-500/40 bg-transparent text-blue-600 dark:text-blue-400";
const CHIP_SAFETENSORS =
  "border-violet-500/40 bg-transparent text-violet-600 dark:text-violet-400";

function QuantBadge({
  quant,
  fit,
  showFit = true,
  active = false,
  variant = "trigger",
}: {
  quant: string;
  fit: FitClass;
  showFit?: boolean;
  active?: boolean;
  variant?: "trigger" | "menu";
}) {
  const meta = FIT_BADGE[fit];
  const inner =
    variant === "menu" ? (
      <span
        className={cn(
          CHIP_BASE,
          "gap-1.5 cursor-help",
          active ? CHIP_ACTIVE : CHIP_DEFAULT,
        )}
      >
        {showFit && (
          <HugeiconsIcon
            icon={InformationCircleIcon}
            strokeWidth={2.25}
            className={cn("size-3.5 shrink-0", meta.iconClassName)}
          />
        )}
        <span>{quant}</span>
      </span>
    ) : (
      <span
        className={cn(
          "inline-flex cursor-help items-center gap-1.5 text-[12.5px] font-medium tracking-tight tabular-nums",
          active ? "text-emerald-600 dark:text-emerald-400" : "text-foreground",
        )}
      >
        {showFit && (
          <HugeiconsIcon
            icon={InformationCircleIcon}
            strokeWidth={2.25}
            className={cn("size-3.5 shrink-0", meta.iconClassName)}
          />
        )}
        <span>{quant}</span>
      </span>
    );
  if (!showFit) return inner;
  return (
    <Tooltip>
      <TooltipTrigger asChild>{inner}</TooltipTrigger>
      <TooltipContent side="top" sideOffset={4}>
        {meta.tooltip}
      </TooltipContent>
    </Tooltip>
  );
}

interface DownloadProgressState {
  variant: string | null;
  expectedBytes: number;
  downloadedBytes: number;
  fraction: number;
}

export function DownloadSection({
  repoId,
  isGguf,
  isDownloaded,
  isActive,
  activeQuant,
  isLoadingThisModel,
  loadingPhase,
  gpuGb,
  systemRamGb,
  onLoad,
  onUseInChat,
  onChange,
}: {
  repoId: string;
  isGguf: boolean;
  isDownloaded: boolean;
  isActive: boolean;
  activeQuant: string | null;
  isLoadingThisModel: boolean;
  loadingPhase?: "downloading" | "starting";
  gpuGb?: number;
  systemRamGb?: number;
  onLoad: (opts: { ggufVariant?: string; expectedBytes?: number }) => void;
  onUseInChat?: () => void;
  onChange?: () => void;
}) {
  const [variants, setVariants] = useState<GgufVariantDetail[] | null>(null);
  const [hasVision, setHasVision] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedQuant, setSelectedQuant] = useState<string | null>(null);
  const [open, setOpen] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [progress, setProgress] = useState<DownloadProgressState | null>(null);
  const [modelTotalBytes, setModelTotalBytes] = useState<number | null>(null);
  const pollRef = useRef<number | null>(null);

  useEffect(() => {
    setModelTotalBytes(null);
    if (isGguf || !repoId) return;
    let cancelled = false;
    void fetchModelSize(repoId).then((info) => {
      if (cancelled || !info) return;
      const upstream = info.weightsBytes ?? info.totalBytes;
      if (upstream && upstream > 0) setModelTotalBytes(upstream);
    });
    return () => {
      cancelled = true;
    };
  }, [isGguf, repoId]);
  const stopPolling = useCallback(() => {
    if (pollRef.current != null) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);
  useEffect(() => {
    return () => stopPolling();
  }, [stopPolling]);

  const refresh = useCallback(() => {
    let canceled = false;
    setLoading(true);
    setError(null);
    listGgufVariants(repoId)
      .then((res) => {
        if (canceled) return;
        setVariants(res.variants);
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
  }, [repoId]);

  useEffect(() => {
    if (!isGguf) {
      setLoading(false);
      return;
    }
    return refresh();
  }, [isGguf, refresh]);

  useEffect(() => {
    setProgress(null);
    setSelectedQuant(null);
    setVariants(null);
    stopPolling();
  }, [repoId, stopPolling]);

  const startDownload = useCallback(
    async (variantOpt: string | null, expectedBytes: number) => {
      stopPolling();
      const initial: DownloadProgressState = {
        variant: variantOpt,
        expectedBytes,
        downloadedBytes: 0,
        fraction: 0,
      };
      setProgress(initial);

      const label = `${repoId}${variantOpt ? ` · ${variantOpt}` : ""}`;
      const toastId = `model-download-${repoId}-${variantOpt ?? ""}`;
      toast(`Downloading ${label}`, {
        id: toastId,
        description: "Starting download…",
        duration: Number.POSITIVE_INFINITY,
      });

      try {
        await startModelDownload({
          repo_id: repoId,
          gguf_variant: variantOpt,
        });
      } catch (err) {
        setProgress(null);
        toast.error("Failed to start download", {
          id: toastId,
          description: err instanceof Error ? err.message : undefined,
          duration: 4000,
        });
        return;
      }

      const tick = async () => {
        try {
          const [progressResp, status] = await Promise.all([
            variantOpt
              ? getGgufDownloadProgress(repoId, variantOpt, expectedBytes)
              : getDownloadProgress(repoId),
            getModelDownloadStatus(repoId, variantOpt),
          ]);

          const nextExpected =
            "expected_bytes" in progressResp
              ? progressResp.expected_bytes
              : expectedBytes;
          const nextProgress: DownloadProgressState = {
            variant: variantOpt,
            expectedBytes: nextExpected,
            downloadedBytes: progressResp.downloaded_bytes,
            fraction: progressResp.progress,
          };
          setProgress(nextProgress);

          const percent = Math.round(
            Math.min(progressResp.progress, 1) * 100,
          );
          const downloadedLabel = formatBytes(progressResp.downloaded_bytes);
          const totalLabel =
            nextExpected > 0 ? formatBytes(nextExpected) : null;
          toast(`Downloading ${label}`, {
            id: toastId,
            description: totalLabel
              ? `${downloadedLabel} / ${totalLabel} · ${percent}%`
              : `${downloadedLabel} · ${percent}%`,
            duration: Number.POSITIVE_INFINITY,
          });

          if (status.state === "complete") {
            stopPolling();
            setProgress(null);
            toast.success(`Downloaded ${label}`, {
              id: toastId,
              description: undefined,
              duration: 3000,
            });
            refresh();
            onChange?.();
          } else if (status.state === "error") {
            stopPolling();
            setProgress(null);
            toast.error("Download failed", {
              id: toastId,
              description: status.error ?? undefined,
              duration: 5000,
            });
            refresh();
          }
        } catch {
          // Transient network errors during a long download are common; the
          // next tick will recover. Don't tear down the UI.
        }
      };

      // Fire one tick immediately so the toast + bar start showing real
      // numbers instead of the 0 B / 0% placeholder, then poll every
      // 500ms — fast enough that small models still get a visible bar.
      void tick();
      pollRef.current = window.setInterval(() => {
        void tick();
      }, 500);
    },
    [repoId, refresh, onChange, stopPolling],
  );

  const recommendedQuant = useMemo(
    () => (variants ? pickRecommendedQuant(variants, gpuGb, systemRamGb) : null),
    [variants, gpuGb, systemRamGb],
  );

  const sortedVariants = useMemo(() => {
    if (!variants) return null;
    const tierOf = (v: GgufVariantDetail) => {
      const f = getFitClass(v.size_bytes, gpuGb, systemRamGb);
      if (f === "oom") return 4;
      const base = f === "fits" ? 0 : 1;
      return v.downloaded ? base : base + 2;
    };
    return [...variants].sort((a, b) => {
      const aTier = tierOf(a);
      const bTier = tierOf(b);
      if (aTier !== bTier) return aTier - bTier;
      const aIsRec = a.quant === recommendedQuant;
      const bIsRec = b.quant === recommendedQuant;
      if (aIsRec !== bIsRec) return aIsRec ? -1 : 1;
      return b.size_bytes - a.size_bytes;
    });
  }, [variants, recommendedQuant, gpuGb, systemRamGb]);

  useEffect(() => {
    if (!sortedVariants || sortedVariants.length === 0) return;
    if (selectedQuant && sortedVariants.some((v) => v.quant === selectedQuant)) {
      return;
    }
    if (activeQuant && sortedVariants.some((v) => v.quant === activeQuant)) {
      setSelectedQuant(activeQuant);
      return;
    }
    const downloaded = sortedVariants.find((v) => v.downloaded);
    if (downloaded) {
      setSelectedQuant(downloaded.quant);
      return;
    }
    if (recommendedQuant && sortedVariants.some((v) => v.quant === recommendedQuant)) {
      setSelectedQuant(recommendedQuant);
      return;
    }
    setSelectedQuant(sortedVariants[0].quant);
  }, [sortedVariants, activeQuant, selectedQuant, recommendedQuant]);

  const selected = sortedVariants?.find((v) => v.quant === selectedQuant) ?? null;

  async function handleDeleteConfirm() {
    if (!deleteTarget) return;
    setDeleting(true);
    try {
      await deleteCachedModel(repoId, deleteTarget);
      toast.success(`Deleted ${repoId} ${deleteTarget}`);
      refresh();
      onChange?.();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to delete");
    } finally {
      setDeleting(false);
      setDeleteTarget(null);
    }
  }

  if (!isGguf) {
    const downloading = progress !== null && progress.variant === null;
    return (
      <div className="download-card">
        <div className="flex items-center">
          <div className="flex h-9 min-w-0 flex-1 items-center pl-3">
            <span className="flex items-center gap-2.5 text-[12px] text-muted-foreground">
              <span className="font-medium text-violet-600 dark:text-violet-400">
                Safetensors
              </span>
              {modelTotalBytes && modelTotalBytes > 0 && (
                <span className="tabular-nums">
                  {formatBytes(modelTotalBytes)}
                </span>
              )}
              {(isActive || isDownloaded) && (
                <span className="inline-flex items-center gap-1 font-medium text-emerald-600 dark:text-emerald-400">
                  <HugeiconsIcon
                    icon={CheckmarkCircle02Icon}
                    strokeWidth={2.5}
                    className="size-3"
                  />
                  {isActive ? "Loaded" : "On device"}
                </span>
              )}
            </span>
          </div>
          <div
            aria-hidden="true"
            className="ml-1 mr-0 h-5 w-px shrink-0 bg-foreground/[0.06] dark:bg-white/[0.04]"
          />
          <button
            type="button"
            disabled={isLoadingThisModel || downloading}
            onClick={() => {
              if (isActive) {
                onUseInChat?.();
                return;
              }
              if (isDownloaded) {
                onLoad({});
              } else {
                void startDownload(null, 0);
              }
            }}
            className={cn(
              "inline-flex h-9 w-24 shrink-0 cursor-pointer items-center justify-center gap-1.5 rounded-[10px] bg-transparent px-3 text-[12.5px] font-medium tracking-tight text-foreground transition-colors hover:bg-foreground/[0.04] dark:hover:bg-white/[0.05]",
              (isLoadingThisModel || downloading) && "opacity-70",
            )}
          >
            {downloading ? (
              <>
                <Spinner className="size-3.5" />
                Downloading…
              </>
            ) : isLoadingThisModel ? (
              <>
                <Spinner className="size-3.5" />
                {loadingPhase === "downloading" ? "Downloading…" : "Loading…"}
              </>
            ) : isActive ? (
              <>
                <HugeiconsIcon
                  icon={PencilEdit02Icon}
                  strokeWidth={1.75}
                  className="size-3.5"
                />
                New Chat
              </>
            ) : isDownloaded ? (
              <>
                <HugeiconsIcon
                  icon={PlayIcon}
                  strokeWidth={1.75}
                  className="size-3.5"
                />
                Run
              </>
            ) : (
              <>
                <HugeiconsIcon
                  icon={Download01Icon}
                  strokeWidth={1.75}
                  className="size-4"
                />
                Download
              </>
            )}
          </button>
        </div>
        {downloading && progress && (
          <DownloadProgressBar progress={progress} />
        )}
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-[12.5px] text-muted-foreground">
        <Spinner className="size-3.5" />
        Loading available quantizations…
      </div>
    );
  }

  if (error || !sortedVariants || sortedVariants.length === 0) {
    return (
      <div className="text-[12.5px] text-destructive">
        {error ?? "No GGUF quantizations found in this repository."}
      </div>
    );
  }

  const downloadingThisVariant =
    progress !== null && progress.variant === selectedQuant;
  const downloadingAnyVariant = progress !== null;
  const ctaDisabled =
    isLoadingThisModel || !selected || downloadingAnyVariant;
  const selectedIsActive =
    isActive && activeQuant && selected?.quant === activeQuant;

  const selectedFit = selected
    ? getFitClass(selected.size_bytes, gpuGb, systemRamGb)
    : null;
  const fitDescription =
    selected && gpuGb && selectedFit ? FIT_BADGE[selectedFit].tooltip : null;
  const fitColor =
    selected && gpuGb && selectedFit
      ? FIT_BADGE[selectedFit].iconClassName
      : "text-muted-foreground";

  return (
    <>
      <div className="download-card">
        <div className="flex items-center">
        <Popover open={open} onOpenChange={setOpen}>
          <PopoverTrigger asChild>
            <button
              type="button"
              className="menu-trigger flex h-9 min-w-0 flex-1 cursor-pointer items-center gap-2.5 rounded-[12px] px-3 text-left transition-colors hover:bg-foreground/[0.04] data-[state=open]:bg-foreground/[0.06] dark:hover:bg-white/[0.04] dark:data-[state=open]:bg-white/[0.06]"
            >
                {selected ? (
                  <QuantBadge
                    quant={selected.quant}
                    fit={getFitClass(
                      selected.size_bytes,
                      gpuGb,
                      systemRamGb,
                    )}
                    showFit={Boolean(gpuGb)}
                    active={Boolean(selectedIsActive)}
                  />
                ) : (
                  <span className="text-[12.5px] text-muted-foreground">
                    Select quantization
                  </span>
                )}
                <span className="ml-auto flex items-center gap-2.5 text-[12px] text-muted-foreground">
                  {selected && !selected.downloaded && (
                    <>
                      <span>GGUF</span>
                      <span className="tabular-nums">
                        {formatBytes(selected.size_bytes)}
                      </span>
                    </>
                  )}
                  <HugeiconsIcon
                    icon={ArrowDown01Icon}
                    strokeWidth={1.25}
                    className="ml-0.5 size-3.5 shrink-0"
                  />
                </span>
              </button>
            </PopoverTrigger>
            <PopoverContent
              align="start"
              side="bottom"
              sideOffset={8}
              avoidCollisions={false}
              noAnimation
              className="menu-instant menu-soft-surface w-[var(--radix-popover-trigger-width)] min-w-[200px] gap-0 overflow-hidden p-0 py-2 ring-0"
            >
              <div className="max-h-[344px] overflow-y-auto [scrollbar-width:thin]">
              {sortedVariants.map((v) => {
                const fit = getFitClass(v.size_bytes, gpuGb, systemRamGb);
                const isSelected = v.quant === selectedQuant;
                const isLoaded = isActive && activeQuant === v.quant;
                return (
                  <div
                    key={v.filename}
                    className={cn(
                      "group relative mx-2 flex items-center gap-2 rounded-[12px] px-2.5 py-2 transition-colors",
                      isSelected
                        ? "bg-foreground/[0.07] dark:bg-foreground/[0.12]"
                        : "hover:bg-foreground/[0.05] dark:hover:bg-foreground/[0.06]",
                    )}
                  >
                    <button
                      type="button"
                      onClick={() => {
                        setSelectedQuant(v.quant);
                        setOpen(false);
                      }}
                      className="flex min-w-0 flex-1 cursor-pointer items-center gap-2 text-left"
                    >
                      <QuantBadge
                        quant={v.quant}
                        fit={fit}
                        active={isLoaded}
                        variant="menu"
                      />
                      {isLoaded && (
                        <span className="text-[10.5px] font-medium text-emerald-600 dark:text-emerald-400">
                          Loaded
                        </span>
                      )}
                    </button>
                    <span className="ml-auto flex shrink-0 items-center gap-1.5">
                      {v.downloaded && (
                        <span className={cn(CHIP_BASE, CHIP_ACTIVE, "gap-1")}>
                          <HugeiconsIcon
                            icon={CheckmarkCircle02Icon}
                            strokeWidth={2.5}
                            className="size-3"
                          />
                          On device
                        </span>
                      )}
                      <span className={cn(CHIP_BASE, CHIP_GGUF)}>GGUF</span>
                      <span className="relative">
                        <span className={cn(CHIP_BASE, CHIP_DEFAULT)}>
                          {formatBytes(v.size_bytes)}
                        </span>
                        {v.downloaded && (
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              setDeleteTarget(v.quant);
                            }}
                            aria-label={`Delete ${v.quant}`}
                            className={cn(
                              "absolute inset-0 inline-flex cursor-pointer items-center justify-center rounded-[7px]",
                              "bg-popover text-foreground/70 ring-1 ring-border transition-colors",
                              "opacity-0 group-hover:opacity-100 focus-visible:opacity-100",
                              "hover:text-destructive hover:ring-destructive/40",
                            )}
                          >
                            <HugeiconsIcon
                              icon={Delete02Icon}
                              strokeWidth={1.75}
                              className="size-3"
                            />
                          </button>
                        )}
                      </span>
                    </span>
                  </div>
                );
              })}
              </div>
            </PopoverContent>
        </Popover>

          <div
            aria-hidden="true"
            className="ml-1 mr-0 h-5 w-px shrink-0 bg-foreground/[0.06] dark:bg-white/[0.04]"
          />

          <button
            type="button"
            disabled={ctaDisabled && !selectedIsActive}
            onClick={() => {
              if (selectedIsActive) {
                onUseInChat?.();
                return;
              }
              if (!selected) return;
              if (selected.downloaded) {
                onLoad({
                  ggufVariant: selected.quant,
                  expectedBytes: selected.size_bytes,
                });
              } else {
                void startDownload(selected.quant, selected.size_bytes);
              }
            }}
            className={cn(
              "inline-flex h-9 w-24 shrink-0 cursor-pointer items-center justify-center gap-1.5 rounded-[10px] bg-transparent px-3 text-[12.5px] font-medium tracking-tight text-foreground transition-colors hover:bg-foreground/[0.04] dark:hover:bg-white/[0.05]",
              ctaDisabled && !selectedIsActive && "opacity-70",
            )}
          >
            {downloadingThisVariant ? (
              <span className="inline-flex items-center gap-2">
                <Spinner className="size-3.5" />
                Downloading…
              </span>
            ) : downloadingAnyVariant ? (
              "Download in progress…"
            ) : isLoadingThisModel ? (
              <span className="inline-flex items-center gap-2">
                <Spinner className="size-3.5" />
                {loadingPhase === "downloading" ? "Downloading…" : "Loading…"}
              </span>
            ) : selectedIsActive ? (
              <>
                <HugeiconsIcon
                  icon={PencilEdit02Icon}
                  strokeWidth={1.75}
                  className="size-3.5"
                />
                New Chat
              </>
            ) : selected?.downloaded ? (
              <>
                <HugeiconsIcon
                  icon={PlayIcon}
                  strokeWidth={1.75}
                  className="size-3.5"
                />
                Run
              </>
            ) : (
              <>
                <HugeiconsIcon
                  icon={Download01Icon}
                  strokeWidth={1.75}
                  className="size-4"
                />
                Download
              </>
            )}
          </button>
        </div>
        {progress !== null && progress.variant === selectedQuant && (
          <DownloadProgressBar progress={progress} />
        )}
      </div>

      <AlertDialog
        open={deleteTarget !== null}
        onOpenChange={(o) => {
          if (!o && !deleting) setDeleteTarget(null);
        }}
      >
        <AlertDialogContent size="sm">
          <AlertDialogHeader>
            <AlertDialogTitle>Delete quantization?</AlertDialogTitle>
            <AlertDialogDescription>
              This will remove{" "}
              <span className="font-medium text-foreground">
                {repoId} ({deleteTarget})
              </span>{" "}
              from disk. You can re-download it later.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={deleting}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              disabled={deleting}
              onClick={(e) => {
                e.preventDefault();
                void handleDeleteConfirm();
              }}
            >
              {deleting ? "Deleting…" : "Delete"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
