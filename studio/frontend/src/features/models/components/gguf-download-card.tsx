// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Spinner } from "@/components/ui/spinner";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  deleteCachedModel,
  getActiveModelDownloads,
  invalidateGgufVariantsCache,
  listGgufVariants,
  type GgufVariantDetail,
} from "@/features/chat";
import { cn } from "@/lib/utils";
import { classifyGgufFit, type GgufFitClass } from "@/lib/gguf-fit";
import {
  ArrowDown01Icon,
  Delete02Icon,
  Download01Icon,
  InformationCircleIcon,
  PencilEdit02Icon,
  PlayIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import { useHfTokenStore } from "@/stores/hf-token-store";
import { formatBytes } from "@/lib/format";
import { useRepoDownload } from "../download-manager";
import {
  CardDivider,
  DeleteConfirmDialog,
  DownloadCard,
} from "./download-card";
import { DotTag } from "./dot-tag";
import { DownloadCancelIndicator } from "./download-cancel-indicator";
import { PathInfoButton } from "./path-info-button";
import type { InventoryHint } from "./download-types";

interface FitBadgeMeta {
  label: string;
  tooltip: string;
  iconClassName: string;
}

const FIT_BADGE: Record<GgufFitClass, FitBadgeMeta> = {
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

/** Chip styling matching the on-device list's StatChip, no icon. */
const CHIP_BASE =
  "inline-flex h-5 shrink-0 items-center justify-center whitespace-nowrap rounded-[7px] border px-1.5 text-[11.5px] font-medium tabular-nums leading-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]";
const CHIP_DEFAULT =
  "border-foreground/15 bg-muted text-foreground/85 dark:border-border/60 dark:bg-white/[0.04] dark:text-foreground/85";
const CHIP_ACTIVE =
  "border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300";

function QuantBadge({
  quant,
  fit,
  showFit = true,
  active = false,
  variant = "trigger",
}: {
  quant: string;
  fit: GgufFitClass;
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

export function GgufDownloadCard({
  repoId,
  isActive,
  activeQuant,
  isLoadingThisModel,
  gpuGb,
  systemRamGb,
  cachePath,
  onLoad,
  onUseInChat,
  onChange,
}: {
  repoId: string;
  isActive: boolean;
  activeQuant: string | null;
  isLoadingThisModel: boolean;
  gpuGb?: number;
  systemRamGb?: number;
  cachePath?: string | null;
  onLoad: (opts: { ggufVariant?: string; expectedBytes?: number }) => void;
  onUseInChat?: () => void;
  onChange?: (hint?: InventoryHint) => void;
}) {
  const hfToken = useHfTokenStore((s) => s.token);
  const variantKey = `${repoId}::${hfToken ?? ""}`;
  const [variantState, setVariantState] = useState<{
    key: string;
    variants: GgufVariantDetail[] | null;
    loading: boolean;
    error: string | null;
  }>(() => ({
    key: variantKey,
    variants: null,
    loading: true,
    error: null,
  }));
  const currentVariantState =
    variantState.key === variantKey
      ? variantState
      : { key: variantKey, variants: null, loading: true, error: null };
  const { variants, loading, error } = currentVariantState;
  const [selectedQuantState, setSelectedQuantState] = useState<{
    repoId: string;
    quant: string | null;
  }>(() => ({ repoId, quant: null }));
  const selectedQuantOverride =
    selectedQuantState.repoId === repoId ? selectedQuantState.quant : null;
  const [open, setOpen] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);

  const mountedRef = useRef(false);
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const refresh = useCallback(
    async (silent = false): Promise<void> => {
      if (silent) {
        invalidateGgufVariantsCache(repoId);
      }
      try {
        const res = await listGgufVariants(repoId, hfToken || undefined);
        if (!mountedRef.current) return;
        setVariantState({
          key: variantKey,
          variants: res.variants,
          loading: false,
          error: null,
        });
      } catch (err) {
        if (!mountedRef.current || silent) return;
        setVariantState({
          key: variantKey,
          variants: null,
          loading: false,
          error: err instanceof Error ? err.message : "Failed to load variants",
        });
      }
    },
    [repoId, hfToken, variantKey],
  );

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const job = useRepoDownload({
    kind: "model",
    repoId,
    onComplete: (variant, bytes) => {
      const done = refresh(true);
      onChange?.({
        kind: variant ? "gguf" : "model",
        repoId,
        bytes: bytes || undefined,
      });
      return done;
    },
    onCancelled: () => {
      const done = refresh(true);
      onChange?.();
      return done;
    },
    onError: () => refresh(true),
  });
  const progress = job.progress;
  const cancelling = job.cancelling;

  const adoptRunningJob = job.adoptRunningJob;
  // Probe once per repo: the silent post-completion refresh swaps the variants
  // array, which would otherwise re-fire the probe needlessly.
  const adoptCheckedRepoRef = useRef<string | null>(null);
  const adoptAbortRef = useRef<AbortController | null>(null);
  useEffect(() => {
    if (!variants) return;
    if (adoptCheckedRepoRef.current === repoId) return;
    adoptCheckedRepoRef.current = repoId;
    adoptAbortRef.current?.abort();
    const controller = new AbortController();
    adoptAbortRef.current = controller;
    void getActiveModelDownloads(repoId, controller.signal)
      .then((downloads) => {
        if (controller.signal.aborted) return;
        const active = downloads.find(
          (d) => d.state === "running" || d.state === "cancelling",
        );
        if (!active) return;
        if (active.variant) {
          setSelectedQuantState({ repoId, quant: active.variant });
        }
        const knownSize = active.variant
          ? (variants.find((v) => v.quant === active.variant)?.size_bytes ?? 0)
          : 0;
        adoptRunningJob(active.variant, knownSize);
      })
      .catch(() => {});
  }, [repoId, variants, adoptRunningJob]);

  useEffect(() => () => adoptAbortRef.current?.abort(), []);

  useEffect(() => {
    adoptAbortRef.current?.abort();
    adoptCheckedRepoRef.current = null;
  }, [repoId]);

  const sortedVariants = useMemo(() => {
    if (!variants) return null;
    const statusRank = (v: GgufVariantDetail) => {
      if (v.downloaded) return 0;
      if (v.partial) return 1;
      return 2;
    };
    const fitRank = (v: GgufVariantDetail): number => {
      switch (classifyGgufFit(v.size_bytes, { gpuGb, systemRamGb })) {
        case "fits":
          return 0;
        case "marginal":
          return 1;
        case "partial":
          return 2;
        default:
          return 3;
      }
    };
    return [...variants].sort((a, b) => {
      const statusDelta = statusRank(a) - statusRank(b);
      if (statusDelta !== 0) return statusDelta;
      const aFit = fitRank(a);
      const bFit = fitRank(b);
      if (aFit !== bFit) return aFit - bFit;
      // Within a tier the larger build is the more capable pick; in the oom
      // tier nothing fits, so the smallest is the least-bad default instead.
      return aFit === 3
        ? a.size_bytes - b.size_bytes
        : b.size_bytes - a.size_bytes;
    });
  }, [variants, gpuGb, systemRamGb]);

  const selectedQuant =
    selectedQuantOverride &&
    sortedVariants?.some((v) => v.quant === selectedQuantOverride)
      ? selectedQuantOverride
      : (sortedVariants?.[0]?.quant ?? null);

  const selected =
    sortedVariants?.find((v) => v.quant === selectedQuant) ?? null;

  async function handleDeleteConfirm() {
    if (!deleteTarget) return;
    setDeleting(true);
    try {
      await deleteCachedModel(repoId, deleteTarget, hfToken || undefined);
      toast.success(`Deleted ${repoId} ${deleteTarget}`);
      refresh(true);
      onChange?.();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to delete");
    } finally {
      setDeleting(false);
      setDeleteTarget(null);
    }
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
  const ctaDisabled = isLoadingThisModel || !selected || downloadingAnyVariant;
  const selectedIsActive =
    isActive && activeQuant && selected?.quant === activeQuant;
  const isGgufRunCta =
    !!selected?.downloaded &&
    !cancelling &&
    !downloadingThisVariant &&
    !downloadingAnyVariant &&
    !isLoadingThisModel &&
    !selectedIsActive;

  return (
    <DownloadCard
      job={job}
      progress={downloadingThisVariant ? progress : null}
      dialogs={
        <DeleteConfirmDialog
          open={deleteTarget !== null}
          onOpenChange={(o) => {
            if (!o && !deleting) setDeleteTarget(null);
          }}
          title="Delete quantization?"
          deleting={deleting}
          onConfirm={() => void handleDeleteConfirm()}
          description={
            <>
              This will remove{" "}
              <span className="font-medium text-foreground">
                {repoId} ({deleteTarget})
              </span>{" "}
              from disk. You can re-download it later.
            </>
          }
        />
      }
    >
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <button
            type="button"
            className="menu-trigger flex h-9 min-w-0 flex-1 cursor-pointer items-center gap-2.5 rounded-[12px] px-3 text-left transition-colors hover:bg-foreground/[0.04] data-[state=open]:bg-foreground/[0.06] dark:hover:bg-white/[0.04] dark:data-[state=open]:bg-white/[0.06]"
          >
            {selected ? (
              <QuantBadge
                quant={selected.quant}
                fit={classifyGgufFit(selected.size_bytes, { gpuGb, systemRamGb })}
                showFit={Boolean(gpuGb)}
                active={Boolean(selectedIsActive)}
              />
            ) : (
              <span className="text-[12.5px] text-muted-foreground">
                Select quantization
              </span>
            )}
            <span className="ml-auto flex items-center gap-1.5 text-[12px] text-muted-foreground">
              {selected?.downloaded && (
                <DotTag
                  tone="success"
                  label={selectedIsActive ? "Loaded" : "On device"}
                />
              )}
              {selected && !selected.downloaded && selected.partial && (
                <Tooltip>
                  <TooltipTrigger asChild>
                    <span className="inline-flex">
                      <DotTag tone="warning" label="Partial" />
                    </span>
                  </TooltipTrigger>
                  <TooltipContent side="top" sideOffset={4}>
                    Partial download. Click to continue.
                  </TooltipContent>
                </Tooltip>
              )}
              <DotTag tone="gguf" label="GGUF" />
              {selected && !selected.downloaded && (
                <span className="tabular-nums">
                  {formatBytes(selected.size_bytes)}
                </span>
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
              const fit = classifyGgufFit(v.size_bytes, { gpuGb, systemRamGb });
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
                      setSelectedQuantState({ repoId, quant: v.quant });
                      setOpen(false);
                    }}
                    className="flex min-w-0 flex-1 cursor-pointer items-center gap-2 text-left"
                  >
                    <QuantBadge
                      quant={v.quant}
                      fit={fit}
                      showFit={Boolean(gpuGb)}
                      active={isLoaded}
                      variant="menu"
                    />
                  </button>
                  <span className="ml-auto flex shrink-0 items-center gap-1.5">
                    {v.downloaded && (
                      <DotTag
                        tone="success"
                        label={isLoaded ? "Loaded" : "On device"}
                      />
                    )}
                    {!v.downloaded && v.partial && (
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <span className="inline-flex">
                            <DotTag tone="warning" label="Partial" />
                          </span>
                        </TooltipTrigger>
                        <TooltipContent side="top" sideOffset={4}>
                          Partial download. Select it to continue.
                        </TooltipContent>
                      </Tooltip>
                    )}
                    <DotTag tone="gguf" label="GGUF" />
                    <span className="relative">
                      <span className={cn(CHIP_BASE, CHIP_DEFAULT)}>
                        {formatBytes(v.size_bytes)}
                      </span>
                      {(v.downloaded || v.partial) && (
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            setDeleteTarget(v.quant);
                          }}
                          aria-label={`Delete ${v.quant}${v.partial && !v.downloaded ? " (partial)" : ""}`}
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

      {selected?.downloaded && cachePath && (
        <PathInfoButton
          path={cachePath}
          title="On-device location"
          description={`Where ${repoId} (${selected.quant}) lives on disk.`}
          className="ml-0.5"
        />
      )}

      {!isGgufRunCta && <CardDivider />}

      <button
        type="button"
        disabled={
          cancelling
            ? true
            : downloadingThisVariant
              ? false
              : ctaDisabled && !selectedIsActive
        }
        onClick={() => {
          if (downloadingThisVariant) {
            void job.cancelDownload(selectedQuant);
            return;
          }
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
            void job.requestStartDownload(selected.quant, selected.size_bytes);
          }
        }}
        aria-label={
          cancelling
            ? "Cancelling…"
            : downloadingThisVariant
              ? "Cancel download"
              : undefined
        }
        className={cn(
          isGgufRunCta ? "run-action-btn w-28" : "hub-action-btn w-28",
          isGgufRunCta && "ml-2",
          ctaDisabled &&
            !selectedIsActive &&
            !downloadingThisVariant &&
            !cancelling &&
            "opacity-70",
          cancelling && "opacity-70",
          downloadingThisVariant &&
            !cancelling &&
            "hover:bg-rose-500/10 hover:text-rose-600 dark:hover:text-rose-400",
        )}
      >
        {cancelling ? (
          <span className="inline-flex items-center gap-2 text-muted-foreground">
            <Spinner />
            Cancelling…
          </span>
        ) : downloadingThisVariant ? (
          <span className="inline-flex items-center gap-2">
            <DownloadCancelIndicator />
            {progress
              ? `${Math.round(Math.min(progress.fraction, 1) * 100)}%`
              : null}
          </span>
        ) : downloadingAnyVariant ? (
          <span className="text-muted-foreground">Busy</span>
        ) : isLoadingThisModel ? (
          <span className="inline-flex items-center gap-2">
            <Spinner />
            Loading…
          </span>
        ) : selectedIsActive ? (
          <>
            <HugeiconsIcon icon={PencilEdit02Icon} strokeWidth={1.75} />
            New Chat
          </>
        ) : selected?.downloaded ? (
          <>
            <HugeiconsIcon icon={PlayIcon} strokeWidth={1.75} />
            Run
          </>
        ) : (
          <>
            <HugeiconsIcon icon={Download01Icon} strokeWidth={1.75} />
            {selected?.partial ? "Continue" : "Download"}
          </>
        )}
      </button>
    </DownloadCard>
  );
}
