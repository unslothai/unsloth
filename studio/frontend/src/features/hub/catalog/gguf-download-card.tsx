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
import {
  type GgufVariantDetail,
  deleteCachedModel,
} from "../inventory";
import { formatBytes } from "../lib/format";
import { type GgufFitClass, classifyGgufFit } from "../lib/gguf-fit";
import { HUB_GGUF_RUN_ACTIONS_VISIBLE } from "../lib/hub-feature-flags";
import {
  ggufVariantsMatch,
  normalizeGgufVariantIdentity,
} from "../lib/model-identity";
import { cn } from "@/lib/utils";
import { useHfTokenStore } from "../stores/hf-token-store";
import { useOnlineStatus } from "../hooks/use-online-status";
import {
  ArrowReloadHorizontalIcon,
  Delete02Icon,
  Download01Icon,
  InformationCircleIcon,
  PencilEdit02Icon,
  PlayIcon,
} from "@hugeicons/core-free-icons";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  memo,
  useCallback,
  useEffect,
  useMemo,
  useState,
  type KeyboardEventHandler,
  type MouseEventHandler,
} from "react";
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
  UpdateConfirmDialog,
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
  "inline-flex h-5 shrink-0 items-center justify-center whitespace-nowrap rounded-full border px-2 text-[11.5px] font-medium tabular-nums leading-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]";
const CHIP_DEFAULT =
  "border-foreground/15 bg-muted text-foreground/85 dark:border-border/60 dark:bg-white/[0.04] dark:text-foreground/85";
const CHIP_ACTIVE =
  "border-control-accent/40 bg-control-accent/10 text-control-accent";

function QuantBadge({
  quant,
  fit,
  showFit = true,
  active = false,
  variant = "trigger",
  tooltipMode = "eager",
}: {
  quant: string;
  fit: GgufFitClass;
  showFit?: boolean;
  active?: boolean;
  variant?: "trigger" | "menu";
  tooltipMode?: "eager" | "lazy" | "none";
}) {
  const meta = FIT_BADGE[fit];
  const [tooltipArmed, setTooltipArmed] = useState(false);
  const [tooltipOpen, setTooltipOpen] = useState(false);
  const armTooltip = useCallback(() => {
    setTooltipArmed((armed) => (armed ? armed : true));
    if (tooltipMode === "lazy") setTooltipOpen(true);
  }, [tooltipMode]);
  const tooltipActive = tooltipMode === "eager" || tooltipArmed;
  const inner =
    variant === "menu" ? (
      <span
        className={cn(
          CHIP_BASE,
          // `shrink` overrides CHIP_BASE's shrink-0 so a long file-path quant
          // label can shrink and truncate instead of overflowing the row.
          "min-w-0 max-w-full shrink gap-1.5 cursor-help",
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
        <span className="min-w-0 truncate">{quant}</span>
      </span>
    ) : (
      // Trigger quant label is the row's primary identity and is short
      // (e.g. "Q4_K_M"); keep it `shrink-0` + `whitespace-nowrap` so it never
      // collapses to "q…" when the Update/Run actions crowd the row. The info
      // group's `overflow-hidden` sacrifices the trailing status tags instead.
      <span
        className={cn(
          "inline-flex shrink-0 cursor-help items-center gap-1.5 whitespace-nowrap text-[12.5px] font-medium tracking-tight tabular-nums",
          active ? "text-control-accent" : "text-foreground",
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
  if (!showFit || tooltipMode === "none") return inner;
  if (!tooltipActive) {
    return (
      <span
        className="inline-flex min-w-0"
        onPointerEnter={armTooltip}
        onFocusCapture={armTooltip}
      >
        {inner}
      </span>
    );
  }
  return (
    <Tooltip
      open={tooltipMode === "lazy" ? tooltipOpen : undefined}
      onOpenChange={tooltipMode === "lazy" ? setTooltipOpen : undefined}
    >
      <TooltipTrigger
        asChild
        onFocusCapture={tooltipMode === "lazy" ? armTooltip : undefined}
        onPointerEnter={tooltipMode === "lazy" ? armTooltip : undefined}
      >
        {inner}
      </TooltipTrigger>
      <TooltipContent side="top" sideOffset={4}>
        {meta.tooltip}
      </TooltipContent>
    </Tooltip>
  );
}

interface GgufVariantMenuItem {
  filename: string;
  key: string;
  quant: string;
  label: string;
  fit: GgufFitClass;
  downloaded: boolean;
  partial: boolean;
  downloadSizeLabel: string;
}

function createGgufVariantMenuItems(
  variants: readonly GgufVariantDetail[] | null,
  resources: { gpuGb?: number; systemRamGb?: number },
): GgufVariantMenuItem[] {
  if (!variants) return [];
  return variants.map((variant) => ({
    filename: variant.filename,
    key: normalizeGgufVariantIdentity(variant.quant),
    quant: variant.quant,
    label: ggufVariantDisplayLabel(variant),
    fit: classifyGgufFit(variant.size_bytes, resources),
    downloaded: Boolean(variant.downloaded),
    partial: Boolean(variant.partial),
    downloadSizeLabel: formatBytes(ggufVariantDownloadSizeBytes(variant)),
  }));
}

const GgufVariantMenuRow = memo(function GgufVariantMenuRow({
  item,
  selected,
  loaded,
  liveActive,
  showFitInfo,
  onSelect,
  onDelete,
}: {
  item: GgufVariantMenuItem;
  selected: boolean;
  loaded: boolean;
  liveActive: boolean;
  showFitInfo: boolean;
  onSelect: (quant: string) => void;
  onDelete: (quant: string) => void;
}) {
  const selectVariant = useCallback(() => {
    onSelect(item.quant);
  }, [item.quant, onSelect]);
  const handleKeyDown = useCallback<KeyboardEventHandler<HTMLDivElement>>(
    (e) => {
      if (e.target !== e.currentTarget) return;
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        selectVariant();
      }
    },
    [selectVariant],
  );
  const handleDelete = useCallback<MouseEventHandler<HTMLButtonElement>>(
    (e) => {
      e.stopPropagation();
      onDelete(item.quant);
    },
    [item.quant, onDelete],
  );
  const canDelete = (item.downloaded || item.partial) && !loaded && !liveActive;

  return (
    <div
      role="button"
      tabIndex={0}
      aria-pressed={selected}
      onClick={selectVariant}
      onKeyDown={handleKeyDown}
      className={cn(
        "group relative mx-2 flex cursor-pointer items-center gap-2 rounded-[12px] px-2.5 py-2 text-left transition-colors",
        selected
          ? "bg-foreground/[0.07] dark:bg-foreground/[0.12]"
          : "hover:bg-foreground/[0.05] dark:hover:bg-foreground/[0.06]",
      )}
    >
      {/* Status (On device / Loaded / Partial) sits beside the quant on the
          left so the model's identity reads as one unit; only the size pins
          right. No per-row "GGUF" tag: every row here is a GGUF quant and the
          trigger already labels it, so repeating it only stole the room the
          quant label needs (it would otherwise truncate to "q…"). */}
      <span className="flex min-w-0 flex-1 items-center gap-2">
        <QuantBadge
          quant={item.label}
          fit={item.fit}
          showFit={showFitInfo}
          active={loaded}
          variant="menu"
          tooltipMode="lazy"
        />
        {item.downloaded && (
          <DotTag tone="success" label={loaded ? "Loaded" : "On device"} />
        )}
        {!item.downloaded && item.partial && (
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
      </span>
      <span className="ml-auto flex shrink-0 items-center gap-1.5">
        <span className="relative">
          <span className={cn(CHIP_BASE, CHIP_DEFAULT)}>
            {item.downloadSizeLabel}
          </span>
          {canDelete && (
            <button
              type="button"
              onClick={handleDelete}
              aria-label={`Delete ${item.label}${item.partial && !item.downloaded ? " (partial)" : ""}`}
              className={cn(
                "absolute inset-0 inline-flex cursor-pointer items-center justify-center rounded-full",
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
});

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
  onChange?: () => void;
}) {
  const hfToken = useHfTokenStore((s) => s.token);
  const online = useOnlineStatus();
  const localVariantPath = cachePath?.trim() || null;
  const { variants, loading, error, refreshError, refresh } =
    useGgufVariantFetchState({
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
  const [updateTarget, setUpdateTarget] = useState<string | null>(null);
  const [completedVariantKeys, setCompletedVariantKeys] = useState<
    ReadonlySet<string>
  >(() => new Set<string>());

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
    const withLive = applyLiveGgufVariantStates(
      rawSortedVariants,
      liveVariantStates,
    );
    if (completedVariantKeys.size === 0) return withLive;
    return withLive.map((v) =>
      completedVariantKeys.has(normalizeGgufVariantIdentity(v.quant))
        ? { ...v, downloaded: true, partial: false, update_available: false }
        : v,
    );
  }, [completedVariantKeys, liveVariantStates, rawSortedVariants]);
  const variantMenuItems = useMemo(
    () => createGgufVariantMenuItems(sortedVariants, { gpuGb, systemRamGb }),
    [gpuGb, sortedVariants, systemRamGb],
  );

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
    onComplete: (variant) => {
      if (!variant) return;
      const key = normalizeGgufVariantIdentity(variant);
      setCompletedVariantKeys((prev) =>
        prev.has(key) ? prev : new Set(prev).add(key),
      );
      void refresh();
    },
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

  useEffect(() => {
    setCompletedVariantKeys(new Set<string>());
  }, [repoId]);

  useEffect(() => {
    if (loading || error || refreshError || !variants) return;
    setCompletedVariantKeys((prev) =>
      prev.size === 0 ? prev : new Set<string>(),
    );
  }, [loading, error, refreshError, variants]);

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
  const selectedFit = useMemo(
    () =>
      selected
        ? classifyGgufFit(selected.size_bytes, { gpuGb, systemRamGb })
        : null,
    [gpuGb, selected?.size_bytes, systemRamGb],
  );
  const selectedDownloadSizeLabel = selected
    ? formatBytes(ggufVariantDownloadSizeBytes(selected))
    : null;
  const updateAvailable =
    selected?.downloaded === true && selected.update_available === true;
  const selectedVariantKey = selectedQuant
    ? normalizeGgufVariantIdentity(selectedQuant)
    : null;
  const activeVariantKey = activeQuant
    ? normalizeGgufVariantIdentity(activeQuant)
    : null;
  const handleSelectVariant = useCallback(
    (quant: string) => {
      setSelectedQuantState({
        repoId,
        quant,
        userPicked: true,
      });
      setOpen(false);
    },
    [repoId],
  );
  const handleDeleteVariant = useCallback((quant: string) => {
    setDeleteTarget(quant);
  }, []);
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
  const updateTargetVariant =
    updateTarget && sortedVariants
      ? sortedVariants.find((v) => ggufVariantsMatch(v.quant, updateTarget))
      : null;
  const updateTargetLabel = updateTargetVariant
    ? ggufVariantDisplayLabel(updateTargetVariant)
    : updateTarget;
  // Confirm → close the dialog and run the re-download as a MANAGED download, so
  // it surfaces in the "Downloading N items" panel with correct manifest-based
  // progress and a working Cancel — the same UX as any other download — instead
  // of a bespoke modal/toast. The worker re-resolves `main` and pulls only the
  // changed blobs, so the cached version stays intact (and runnable) until the
  // new revision lands. Completion refreshes the variant list, whose metadata
  // carries the "Update available" cue.
  const handleConfirmUpdate = useCallback(() => {
    if (!updateTarget) return;
    const variant = updateTarget;
    const expectedBytes =
      updateTargetVariant?.download_size_bytes ??
      updateTargetVariant?.size_bytes ??
      0;
    setUpdateTarget(null);
    void downloadManager.requestStart({
      kind: "model",
      repoId,
      variant,
      expectedBytes,
    });
  }, [updateTarget, updateTargetVariant, repoId]);
  const variantListUnavailable = !sortedVariants || sortedVariants.length === 0;
  const showVariantLoadingState = loading && variantListUnavailable;

  // Keep showing download progress while the variant list is unavailable, so a
  // remount never hides an in-flight download behind the variant status card.
  if (progress && variantListUnavailable) {
    return (
      <GgufDownloadingFallbackCard
        job={job}
        progress={progress}
        cancelling={cancelling}
      />
    );
  }

  if (showVariantLoadingState) {
    return (
      <GgufDownloadStatusCard
        job={job}
        loading
        message="Loading available quantizations…"
      />
    );
  }

  if (variantListUnavailable) {
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
          <>
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
            <UpdateConfirmDialog
              open={updateTarget !== null}
              onOpenChange={(o) => {
                if (!o) setUpdateTarget(null);
              }}
              title="Update quantization?"
              updating={false}
              onConfirm={handleConfirmUpdate}
              description={
                <>
                  This will re-download the latest version of{" "}
                  <span className="font-medium text-foreground">
                    {repoId} ({updateTargetLabel})
                  </span>{" "}
                  from Hugging Face.
                </>
              }
            />
          </>
        }
      >
        <Popover open={open} onOpenChange={setOpen}>
          <PopoverTrigger asChild>
            <button
              type="button"
              onClick={(e) => {
                e.preventDefault();
                setOpen((o) => !o);
              }}
              className="hub-menu-trigger flex h-9 min-w-0 flex-1 cursor-pointer items-center gap-2 rounded-full px-3 text-left transition-colors hover:bg-foreground/[0.04] data-[state=open]:bg-foreground/[0.06] dark:hover:bg-white/[0.04] dark:data-[state=open]:bg-white/[0.06]"
            >
              {/* Quant label + status tags travel together as one left-aligned
                  group so the fit-info icon never floats orphaned from its tags;
                  only the chevron pins right, the standard select affordance. */}
              <span className="flex min-w-0 flex-1 items-center gap-2 overflow-hidden text-[12px] text-muted-foreground">
                {selected ? (
                  <QuantBadge
                    quant={selectedLabel ?? selected.quant}
                    fit={selectedFit ?? "oom"}
                    showFit={showFitInfo}
                    active={Boolean(selectedIsActive)}
                  />
                ) : (
                  <span className="text-[12.5px] text-muted-foreground">
                    Select quantization
                  </span>
                )}
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
                {selected &&
                  selectedDownloadSizeLabel &&
                  !selected.downloaded && (
                    <span className="shrink-0 tabular-nums">
                      {selectedDownloadSizeLabel}
                    </span>
                  )}
              </span>
              <HugeiconsIcon
                icon={ChevronDownStandardIcon}
                className="size-3.5 shrink-0 text-muted-foreground"
              />
            </button>
          </PopoverTrigger>
          <PopoverContent
            align="start"
            side="bottom"
            sideOffset={8}
            avoidCollisions={false}
            className="hub-menu-instant menu-soft-surface w-[var(--radix-popover-trigger-width)] min-w-[300px] gap-0 overflow-hidden p-0 py-2 ring-0"
          >
            <div className="max-h-[344px] overflow-y-auto [scrollbar-width:thin]">
              {variantMenuItems.map((item) => {
                const liveState = liveVariantStates.get(item.key);
                const liveActive = activeDownloadState(liveState?.state);
                return (
                  <GgufVariantMenuRow
                    key={item.filename}
                    item={item}
                    selected={item.key === selectedVariantKey}
                    loaded={isActive && item.key === activeVariantKey}
                    liveActive={liveActive}
                    showFitInfo={showFitInfo}
                    onSelect={handleSelectVariant}
                    onDelete={handleDeleteVariant}
                  />
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

        {selected?.downloaded &&
          online &&
          updateAvailable &&
          !selectedIsActive &&
          !downloadingThisVariant && (
            <button
              type="button"
              onClick={() => selected && setUpdateTarget(selected.quant)}
              aria-label={`Update ${repoId}`}
              className="hub-action-btn ml-1 text-amber-700 dark:text-amber-300"
            >
              <HugeiconsIcon
                icon={ArrowReloadHorizontalIcon}
                strokeWidth={1.75}
              />
              Update
            </button>
          )}

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
            !HUB_GGUF_RUN_ACTIONS_VISIBLE &&
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
      {refreshError && (
        <button
          type="button"
          onClick={() => void refresh()}
          className="self-start px-1 text-[11px] text-status-warning underline-offset-2 transition-colors hover:underline"
        >
          Couldn't refresh quantizations. Retry
        </button>
      )}
    </div>
  );
}
