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
  downloadManager,
  useDownloadManagerStore,
  useRepoDownload,
} from "../download-manager";
import { deleteCachedModel } from "../inventory";
import { formatBytes } from "../lib/format";
import { type GgufFitClass, classifyGgufFit } from "../lib/gguf-fit";
import { HUB_POST_DOWNLOAD_ACTIONS_VISIBLE } from "../lib/hub-feature-flags";
import {
  ggufVariantsMatch,
  normalizeGgufVariantIdentity,
} from "../lib/model-identity";
import { cn } from "@/lib/utils";
import { useHfTokenStore } from "../stores/hf-token-store";
import {
  ArrowDown01Icon,
  Delete02Icon,
  Download01Icon,
  InformationCircleIcon,
  PencilEdit02Icon,
  PlayIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useMemo, useState } from "react";
import {
  ggufVariantDisplayLabel,
  ggufVariantDownloadSizeBytes,
  sortDownloadableGgufVariants,
} from "../lib/gguf-variant-sort";
import { DotTag } from "./dot-tag";
import { DownloadCancelIndicator } from "./download-cancel-indicator";
import {
  CardDivider,
  DeleteConfirmDialog,
  DownloadCard,
} from "./download-card";
import {
  activeDownloadState,
  applyLiveGgufVariantStates,
  createLiveGgufVariantStatesSelector,
} from "./gguf-live-variant-states";
import {
  GgufDownloadStatusCard,
  GgufDownloadingFallbackCard,
} from "./gguf-status-cards";
import type { InventoryHint } from "../inventory";
import { PathInfoButton } from "./path-info-button";
import { useDeleteConfirmAction } from "./use-delete-confirm-action";
import { useDownloadCardState } from "./use-download-card-state";
import { useGgufVariantFetchState } from "./use-gguf-variant-fetch-state";

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
  ram: {
    label: "RAM fallback",
    tooltip:
      "No GPU VRAM detected. This GGUF may run with system RAM and CPU offload. Inference will be slower.",
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
  preferLocalCache = false,
  isPartial = false,
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
  preferLocalCache?: boolean;
  isPartial?: boolean;
  onLoad: (opts: { ggufVariant?: string; expectedBytes?: number }) => void;
  onUseInChat?: () => void;
  onChange?: (hint?: InventoryHint) => void;
}) {
  const hfToken = useHfTokenStore((s) => s.token);
  const localVariantPath = cachePath?.trim() || null;
  const { variants, loading, error, refresh } = useGgufVariantFetchState({
    repoId,
    hfToken,
    preferLocalCache,
    localPath: localVariantPath,
  });
  const [selectedQuantState, setSelectedQuantState] = useState<{
    repoId: string;
    quant: string | null;
    userPicked?: boolean;
  }>(() => ({ repoId, quant: null }));
  const selectedQuantOverride =
    selectedQuantState.repoId === repoId ? selectedQuantState.quant : null;
  const [open, setOpen] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);

  const rawSortedVariants = useMemo(() => {
    if (!variants) return null;
    return sortDownloadableGgufVariants(variants, { gpuGb, systemRamGb });
  }, [variants, gpuGb, systemRamGb]);
  const selectLiveGgufVariantStates = useMemo(
    () => createLiveGgufVariantStatesSelector(repoId),
    [repoId],
  );
  const liveVariantStates = useDownloadManagerStore(selectLiveGgufVariantStates);
  const sortedVariants = useMemo(() => {
    if (!rawSortedVariants) return null;
    return applyLiveGgufVariantStates(rawSortedVariants, liveVariantStates);
  }, [liveVariantStates, rawSortedVariants]);

  const selectedQuant =
    (selectedQuantOverride
      ? sortedVariants?.find((v) =>
          ggufVariantsMatch(v.quant, selectedQuantOverride),
        )?.quant
      : null) ??
    sortedVariants?.[0]?.quant ??
    null;

  const job = useRepoDownload({
    kind: "model",
    repoId,
    activeVariant: selectedQuant ?? undefined,
  });
  const progress = job.progress;
  const cancelling = job.cancelling;

  const setExpectedBytes = job.setExpectedBytes;
  useEffect(() => {
    const controller = new AbortController();
    void downloadManager.probeAndAdopt("model", repoId, controller.signal, {
      includeVariants: true,
      fresh: true,
      onModelAdopt: (active) => {
        if (controller.signal.aborted) return;
        if (active.variant) {
          setSelectedQuantState((prev) =>
            prev.repoId === repoId && prev.userPicked
              ? prev
              : { repoId, quant: active.variant },
          );
        }
      },
    });
    return () => {
      controller.abort();
    };
  }, [repoId]);

  useEffect(() => {
    if (!variants || !progress?.variant) return;
    const knownVariant = variants.find((variant) =>
      ggufVariantsMatch(variant.quant, progress.variant),
    );
    if (!knownVariant) return;
    const expectedBytes =
      knownVariant.download_size_bytes ?? knownVariant.size_bytes ?? 0;
    if (expectedBytes > progress.expectedBytes) {
      setExpectedBytes(expectedBytes, progress.variant);
    }
  }, [
    variants,
    progress?.variant,
    progress?.expectedBytes,
    setExpectedBytes,
  ]);

  const selected =
    sortedVariants?.find((v) => ggufVariantsMatch(v.quant, selectedQuant)) ??
    null;
  const selectedLiveState = selectedQuant
    ? liveVariantStates.get(normalizeGgufVariantIdentity(selectedQuant))
    : undefined;
  const selectedLiveActive = activeDownloadState(selectedLiveState?.state);
  const downloadingThisVariant =
    progress !== null && ggufVariantsMatch(progress.variant, selectedQuant);
  const ctaDisabled = isLoadingThisModel || !selected;
  const selectedIsActive =
    isActive && activeQuant && ggufVariantsMatch(selected?.quant, activeQuant);
  const isGgufRunCta =
    !!selected?.downloaded &&
    !cancelling &&
    !downloadingThisVariant &&
    !isLoadingThisModel &&
    !selectedIsActive;
  const showFitInfo = Boolean(gpuGb) || Boolean(systemRamGb);
  const downloadAction = useDownloadCardState({
    job,
    variant: selectedQuant,
    expectedBytes: selected?.download_size_bytes ?? selected?.size_bytes ?? 0,
    downloading: downloadingThisVariant,
    cancelling,
    disabled: cancelling
      ? true
      : downloadingThisVariant
        ? false
        : ctaDisabled && !selectedIsActive,
    isPartial: Boolean(selected?.partial),
    partialTransport: selected?.partial_transport ?? null,
  });
  const selectedLabel = selected ? ggufVariantDisplayLabel(selected) : null;
  const deleteTargetVariant =
    deleteTarget && sortedVariants
      ? sortedVariants.find((v) => ggufVariantsMatch(v.quant, deleteTarget))
      : null;
  const deleteTargetLabel = deleteTargetVariant
    ? ggufVariantDisplayLabel(deleteTargetVariant)
    : deleteTarget;
  const { deleting, runDelete } = useDeleteConfirmAction({
    action: async () => {
      if (!deleteTarget) return;
      await deleteCachedModel(repoId, deleteTarget, hfToken || undefined);
    },
    successMessage: () => `Deleted ${repoId} ${deleteTargetLabel ?? deleteTarget}`,
    errorToast: (err) => ({
      title: err instanceof Error ? err.message : "Failed to delete",
    }),
    onSuccess: () => {
      onChange?.();
    },
    onSettled: () => {
      setDeleteTarget(null);
    },
  });

  // A live download for this repo must keep showing progress even while the
  // variant list is still loading, failed to load, or cannot identify the
  // running variant. A remount (On Device tab, page refresh) must never hide an
  // in-flight download behind the variant status card.
  if (
    progress &&
    (loading || error || !sortedVariants || sortedVariants.length === 0)
  ) {
    return (
      <GgufDownloadingFallbackCard
        job={job}
        progress={progress}
        cancelling={cancelling}
      />
    );
  }

  if (loading) {
    return (
      <GgufDownloadStatusCard
        job={job}
        loading
        message="Loading available quantizations…"
      />
    );
  }

  if (error || !sortedVariants || sortedVariants.length === 0) {
    if (isPartial) {
      return (
        <GgufDownloadStatusCard
          job={job}
          tone="muted"
          partial
          message="Partial download present. Couldn't load quantizations."
          actionLabel="Reload"
          onAction={() => void refresh()}
        />
      );
    }
    return (
      <GgufDownloadStatusCard
        job={job}
        tone="danger"
        message={error ?? "No GGUF quantizations found in this repository."}
      />
    );
  }

  return (
    <div className="flex w-full flex-col gap-2">
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
            onConfirm={() => void runDelete()}
            description={
              <>
                This will remove{" "}
                <span className="font-medium text-foreground">
                  {repoId} ({deleteTargetLabel})
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
              className="hub-menu-trigger flex h-9 min-w-0 flex-1 cursor-pointer items-center gap-2.5 rounded-[12px] px-3 text-left transition-colors hover:bg-foreground/[0.04] data-[state=open]:bg-foreground/[0.06] dark:hover:bg-white/[0.04] dark:data-[state=open]:bg-white/[0.06]"
            >
              {selected ? (
                <QuantBadge
                  quant={selectedLabel ?? selected.quant}
                  fit={classifyGgufFit(selected.size_bytes, {
                    gpuGb,
                    systemRamGb,
                  })}
                  showFit={showFitInfo}
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
                        <DotTag
                          tone="warning"
                          label={selectedLiveActive ? "Downloading" : "Partial"}
                        />
                      </span>
                    </TooltipTrigger>
                    <TooltipContent side="top" sideOffset={4}>
                      {selectedLiveActive
                        ? "Download is running. Click to cancel."
                        : "Partial download. Click to continue."}
                    </TooltipContent>
                  </Tooltip>
                )}
                <DotTag tone="gguf" label="GGUF" />
                {selected && !selected.downloaded && (
                  <span className="tabular-nums">
                    {formatBytes(ggufVariantDownloadSizeBytes(selected))}
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
            className="hub-menu-instant menu-soft-surface w-[var(--radix-popover-trigger-width)] min-w-[200px] gap-0 overflow-hidden p-0 py-2 ring-0"
          >
            <div className="max-h-[344px] overflow-y-auto [scrollbar-width:thin]">
              {sortedVariants.map((v) => {
                const label = ggufVariantDisplayLabel(v);
                const fit = classifyGgufFit(v.size_bytes, {
                  gpuGb,
                  systemRamGb,
                });
                const isSelected = ggufVariantsMatch(v.quant, selectedQuant);
                const isLoaded =
                  isActive && ggufVariantsMatch(activeQuant, v.quant);
                const liveState = liveVariantStates.get(
                  normalizeGgufVariantIdentity(v.quant),
                );
                const liveActive = activeDownloadState(liveState?.state);
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
                        setSelectedQuantState({
                          repoId,
                          quant: v.quant,
                          userPicked: true,
                        });
                        setOpen(false);
                      }}
                      className="flex min-w-0 flex-1 cursor-pointer items-center gap-2 text-left"
                    >
                      <QuantBadge
                        quant={label}
                        fit={fit}
                        showFit={showFitInfo}
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
                              <DotTag
                                tone="warning"
                                label={liveActive ? "Downloading" : "Partial"}
                              />
                            </span>
                          </TooltipTrigger>
                          <TooltipContent side="top" sideOffset={4}>
                            {liveActive
                              ? "Download is running. Select it to view progress."
                              : "Partial download. Select it to continue."}
                          </TooltipContent>
                        </Tooltip>
                      )}
                      <DotTag tone="gguf" label="GGUF" />
                      <span className="relative">
                        <span className={cn(CHIP_BASE, CHIP_DEFAULT)}>
                          {formatBytes(ggufVariantDownloadSizeBytes(v))}
                        </span>
                        {(v.downloaded || v.partial) && !isLoaded && !liveActive && (
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              setDeleteTarget(v.quant);
                            }}
                            aria-label={`Delete ${label}${v.partial && !v.downloaded ? " (partial)" : ""}`}
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
            description={`Where ${repoId} (${selectedLabel}) lives on disk.`}
            className="ml-0.5"
          />
        )}

        {!isGgufRunCta && <CardDivider />}

        <button
          type="button"
          disabled={downloadAction.disabled}
          onClick={() => {
            if (downloadingThisVariant) {
              downloadAction.onClick();
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
              downloadAction.onClick();
            }
          }}
          aria-label={downloadAction.ariaLabel}
          className={cn(
            isGgufRunCta ? "hub-run-action-btn w-28" : "hub-action-btn w-28",
            isGgufRunCta && "ml-2",
            ctaDisabled &&
              !selectedIsActive &&
              !downloadingThisVariant &&
              !cancelling &&
              "opacity-70",
            (cancelling || downloadAction.starting) && "opacity-70",
            downloadingThisVariant &&
              !cancelling &&
              "hover:bg-rose-500/10 hover:text-rose-600 dark:hover:text-rose-400",
            // Hide post-download CTAs (Run / New Chat) for this PR.
            !HUB_POST_DOWNLOAD_ACTIONS_VISIBLE &&
              !downloadingThisVariant &&
              !cancelling &&
              !isLoadingThisModel &&
              (selectedIsActive || selected?.downloaded) &&
              "hidden",
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
              {downloadAction.progressPercent != null
                ? `${downloadAction.progressPercent}%`
                : null}
            </span>
          ) : downloadAction.starting ? (
            <span className="inline-flex items-center gap-2">
              <Spinner />
              Starting…
            </span>
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
              {downloadAction.downloadLabel}
            </>
          )}
        </button>
      </DownloadCard>
    </div>
  );
}
