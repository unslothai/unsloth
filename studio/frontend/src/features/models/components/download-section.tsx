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
  Alert02Icon,
  ArrowDataTransferHorizontalIcon,
  ArrowDown01Icon,
  Cancel01Icon,
  CheckmarkCircle02Icon,
  Delete02Icon,
  DownloadCircle02Icon,
  PlayIcon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
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
          className="h-full rounded-full bg-foreground/80 transition-[width] duration-300"
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
  icon: IconSvgElement;
  label: string;
  tooltip: string;
  className: string;
  iconClassName: string;
}

const FIT_BADGE: Record<FitClass, FitBadgeMeta> = {
  fits: {
    icon: Tick02Icon,
    label: "Full GPU offload",
    tooltip: "Full offload possible. Fits comfortably in VRAM.",
    className:
      "bg-primary border-primary/30 dark:border-primary/40",
    iconClassName: "text-primary-foreground",
  },
  marginal: {
    icon: Alert02Icon,
    label: "Might fit",
    tooltip:
      "Might fit. Within the last GB of VRAM headroom, so loading can fail if other apps are using GPU memory.",
    className:
      "bg-orange-500 border-orange-600/30 dark:bg-orange-500 dark:border-orange-400/30",
    iconClassName: "text-white",
  },
  partial: {
    icon: ArrowDataTransferHorizontalIcon,
    label: "Partial offload",
    tooltip:
      "Partial offload possible. Exceeds VRAM but fits with system RAM offload. Inference will be slower.",
    className:
      "bg-blue-500 border-blue-600/25 dark:bg-blue-500 dark:border-blue-400/25",
    iconClassName: "text-white",
  },
  oom: {
    icon: Cancel01Icon,
    label: "Won't fit",
    tooltip: "Won't fit. Exceeds combined VRAM and system RAM budget.",
    className:
      "bg-rose-400 border-rose-500/30 dark:bg-rose-400 dark:border-rose-300/30",
    iconClassName: "text-white",
  },
};

function FitBadge({
  fit,
  size = "md",
}: {
  fit: FitClass;
  size?: "sm" | "md";
}) {
  const meta = FIT_BADGE[fit];
  return (
    <Tooltip>
      <TooltipTrigger
        type="button"
        aria-label={meta.label}
        className={cn(
          "inline-flex shrink-0 items-center justify-center rounded-[6px] border transition-colors",
          size === "sm" ? "size-[18px]" : "size-5",
          meta.className,
        )}
      >
        <HugeiconsIcon
          icon={meta.icon}
          strokeWidth={2}
          className={cn(
            size === "sm" ? "size-3" : "size-3.5",
            meta.iconClassName,
          )}
        />
      </TooltipTrigger>
      <TooltipContent side="top" sideOffset={4}>
        {meta.tooltip}
      </TooltipContent>
    </Tooltip>
  );
}

function QuantChip({
  label,
  size = "md",
  tone = "default",
}: {
  label: string;
  size?: "sm" | "md";
  tone?: "default" | "active";
}) {
  return (
    <span
      className={cn(
        "inline-flex shrink-0 items-center justify-center rounded-[6px] border font-mono font-semibold tracking-tight",
        size === "sm" ? "h-5 px-1.5 text-[10px]" : "h-6 px-2 text-[11px]",
        tone === "active"
          ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
          : "border-border/60 bg-muted/60 text-foreground",
      )}
    >
      {label}
    </span>
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
  const pollRef = useRef<number | null>(null);
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
      <div className="flex flex-col gap-3 rounded-[24px] border border-border/60 bg-background/80 p-4">
        <div className="flex items-center justify-between gap-3">
          <div className="flex min-w-0 flex-col gap-0.5">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
              Download
            </span>
            <span className="text-[13px] text-foreground">
              {isDownloaded
                ? "Already downloaded — ready to load into memory."
                : downloading
                  ? "Downloading model weights to local cache."
                  : "Pulls the full safetensors snapshot to your HF cache."}
            </span>
          </div>
          <button
            type="button"
            disabled={isLoadingThisModel || isActive || downloading}
            onClick={() => {
              if (isDownloaded) {
                onLoad({});
              } else {
                void startDownload(null, 0);
              }
            }}
            className={cn(
              "inline-flex h-10 shrink-0 items-center gap-1.5 rounded-[16px] px-4 text-[12.5px] font-medium transition-colors",
              isActive
                ? "cursor-default bg-emerald-500/10 text-emerald-700 dark:text-emerald-400"
                : "bg-foreground text-background hover:bg-foreground/85",
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
              "Loaded"
            ) : isDownloaded ? (
              <>
                <HugeiconsIcon icon={PlayIcon} strokeWidth={1.75} className="size-3.5" />
                Load
              </>
            ) : (
              <>
                <HugeiconsIcon
                  icon={DownloadCircle02Icon}
                  strokeWidth={1.75}
                  className="size-3.5"
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
      <div className="flex items-center gap-2 rounded-[24px] border border-border/60 bg-background/80 px-4 py-4 text-[12.5px] text-muted-foreground">
        <Spinner className="size-3.5" />
        Loading available quantizations…
      </div>
    );
  }

  if (error || !sortedVariants || sortedVariants.length === 0) {
    return (
      <div className="rounded-[24px] border border-destructive/30 bg-destructive/5 px-4 py-3 text-[12.5px] text-destructive">
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

  return (
    <>
      <div className="flex flex-col gap-3 rounded-[24px] border border-border/60 bg-background/80 p-4">
        <div className="flex items-center justify-between gap-2">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
            Download
          </span>
          <span className="text-[11px] text-muted-foreground">
            {sortedVariants.length} quantization
            {sortedVariants.length === 1 ? "" : "s"}
            {hasVision && (
              <span className="ml-2 inline-flex h-5 items-center rounded-full border border-violet-500/40 px-2 text-[9.5px] font-semibold uppercase tracking-wider text-violet-600 dark:text-violet-400">
                Vision
              </span>
            )}
          </span>
        </div>

        <div className="flex items-center gap-2">
          <Popover open={open} onOpenChange={setOpen}>
            <PopoverTrigger asChild>
              <button
                type="button"
                className={cn(
                  "flex h-10 min-w-0 flex-1 items-center gap-2.5 rounded-[16px] border border-border/60 bg-background px-3 text-left transition-colors hover:bg-muted/40",
                  open && "ring-2 ring-ring/30",
                )}
              >
                {selected ? (
                  <>
                    <QuantChip
                      label={selected.quant}
                      tone={selectedIsActive ? "active" : "default"}
                    />
                    <span className="flex min-w-0 flex-col leading-tight">
                      {selected.downloaded && (
                        <span className="truncate text-[12.5px] font-medium text-foreground">
                          Downloaded
                        </span>
                      )}
                      <span className="text-[11px] text-muted-foreground">
                        {formatBytes(selected.size_bytes)}
                        {(() => {
                          if (!gpuGb) return null;
                          const fit = getFitClass(
                            selected.size_bytes,
                            gpuGb,
                            systemRamGb,
                          );
                          return ` · ${FIT_BADGE[fit].label.toLowerCase()}`;
                        })()}
                      </span>
                    </span>
                    {gpuGb && (
                      <FitBadge
                        fit={getFitClass(
                          selected.size_bytes,
                          gpuGb,
                          systemRamGb,
                        )}
                      />
                    )}
                  </>
                ) : (
                  <span className="text-[12.5px] text-muted-foreground">
                    Select quantization
                  </span>
                )}
                <HugeiconsIcon
                  icon={ArrowDown01Icon}
                  strokeWidth={1.75}
                  className="ml-auto size-4 shrink-0 text-muted-foreground"
                />
              </button>
            </PopoverTrigger>
            <PopoverContent
              align="start"
              sideOffset={6}
              className="menu-soft-surface w-[var(--radix-popover-trigger-width)] min-w-[280px] max-h-[360px] overflow-y-auto p-1 ring-0"
            >
              {sortedVariants.map((v) => {
                const fit = getFitClass(v.size_bytes, gpuGb, systemRamGb);
                const isSelected = v.quant === selectedQuant;
                const isLoaded = isActive && activeQuant === v.quant;
                return (
                  <div
                    key={v.filename}
                    className={cn(
                      "group flex items-center gap-2 rounded-[8px] px-2 py-1.5 transition-colors",
                      isSelected ? "bg-muted/70" : "hover:bg-muted/50",
                    )}
                  >
                    <button
                      type="button"
                      onClick={() => {
                        setSelectedQuant(v.quant);
                        setOpen(false);
                      }}
                      className="flex min-w-0 flex-1 items-center gap-2.5 text-left"
                    >
                      <QuantChip
                        label={v.quant}
                        size="sm"
                        tone={isLoaded ? "active" : "default"}
                      />
                      <span className="flex min-w-0 flex-1 items-center gap-1.5">
                        <span className="text-[11px] tabular-nums text-muted-foreground">
                          {formatBytes(v.size_bytes)}
                        </span>
                        {v.downloaded && (
                          <HugeiconsIcon
                            icon={CheckmarkCircle02Icon}
                            strokeWidth={2}
                            className="size-3 text-emerald-500"
                          />
                        )}
                        <FitBadge fit={fit} size="sm" />
                        {isLoaded && (
                          <span className="ml-auto text-[10px] font-medium text-emerald-600 dark:text-emerald-400">
                            Loaded
                          </span>
                        )}
                      </span>
                    </button>
                    {v.downloaded && (
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          setDeleteTarget(v.quant);
                        }}
                        aria-label={`Delete ${v.quant}`}
                        className="rounded-md p-1 text-muted-foreground/60 opacity-0 transition-colors group-hover:opacity-100 hover:bg-destructive/10 hover:text-destructive"
                      >
                        <HugeiconsIcon
                          icon={Delete02Icon}
                          strokeWidth={1.75}
                          className="size-3"
                        />
                      </button>
                    )}
                  </div>
                );
              })}
            </PopoverContent>
          </Popover>

          <button
            type="button"
            disabled={ctaDisabled || Boolean(selectedIsActive)}
            onClick={() => {
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
              "inline-flex h-10 shrink-0 items-center gap-1.5 rounded-[16px] px-4 text-[12.5px] font-medium transition-colors",
              selectedIsActive
                ? "cursor-default bg-emerald-500/10 text-emerald-700 dark:text-emerald-400"
                : "bg-foreground text-background hover:bg-foreground/85",
              ctaDisabled && !selectedIsActive && "opacity-70",
            )}
          >
            {downloadingThisVariant ? (
              <>
                <Spinner className="size-3.5" />
                Downloading…
              </>
            ) : downloadingAnyVariant ? (
              "Download in progress…"
            ) : isLoadingThisModel ? (
              <>
                <Spinner className="size-3.5" />
                {loadingPhase === "downloading" ? "Downloading…" : "Loading…"}
              </>
            ) : selectedIsActive ? (
              "Loaded"
            ) : selected?.downloaded ? (
              <>
                <HugeiconsIcon icon={PlayIcon} strokeWidth={1.75} className="size-3.5" />
                Load
              </>
            ) : (
              <>
                <HugeiconsIcon
                  icon={DownloadCircle02Icon}
                  strokeWidth={1.75}
                  className="size-3.5"
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
