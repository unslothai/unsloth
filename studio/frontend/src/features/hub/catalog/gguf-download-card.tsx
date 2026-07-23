// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
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
import { usePlatformStore } from "@/config/env";
import { getCachedModelPath, revealCachedModel } from "@/features/chat";
import { pinKey, usePinnedModelsStore } from "@/features/model-picker";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  ArrowReloadHorizontalIcon,
  Copy01Icon,
  Delete02Icon,
  Download01Icon,
  Folder01Icon,
  InformationCircleIcon,
  MoreVerticalIcon,
  PinIcon,
  PinOffIcon,
  PlayIcon,
  RemoveCircleIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type KeyboardEventHandler,
  memo,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";
import {
  downloadManager,
  useDownloadManagerStore,
  useRepoDownload,
} from "../download-manager";
import { useOnlineStatus } from "../hooks/use-online-status";
import { type GgufVariantDetail, deleteCachedModel } from "../inventory";
import { formatBytes } from "../lib/format";
import { type GgufFitClass, classifyGgufFit } from "../lib/gguf-fit";
import {
  ggufVariantDisplayLabel,
  ggufVariantDownloadSizeBytes,
  sortDownloadableGgufVariants,
} from "../lib/gguf-variant-sort";
import { HUB_GGUF_RUN_ACTIONS_VISIBLE } from "../lib/hub-feature-flags";
import {
  ggufVariantsMatch,
  normalizeGgufVariantIdentity,
} from "../lib/model-identity";
import { useHfTokenStore } from "../stores/hf-token-store";
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
  "inline-flex h-5 shrink-0 items-center justify-center whitespace-nowrap rounded-full border px-2 text-[0.71875rem] font-medium tabular-nums leading-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]";
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
          "inline-flex shrink-0 cursor-help items-center gap-1.5 whitespace-nowrap text-[0.78125rem] font-medium tracking-tight tabular-nums",
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
        asChild={true}
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

// Shared options menu: used on every variant row, the run bar, and the
// single-model (non-GGUF) run bar. Omit `quant` for a repo-level model. The
// identifier uses llama.cpp's repo:quant syntax so it pastes into `-hf`.
export function QuantOptionsMenu({
  repoId,
  quant,
  label,
  downloaded,
  canDelete,
  onDelete,
  showPin = true,
  buttonClassName,
  iconClassName,
}: {
  repoId: string;
  quant?: string;
  label: string;
  downloaded: boolean;
  canDelete: boolean;
  onDelete: (quant?: string) => void;
  // Hidden in the run bar; pinning belongs to the On Device list.
  showPin?: boolean;
  buttonClassName?: string;
  iconClassName?: string;
}) {
  const pinnedKeys = usePinnedModelsStore((s) => s.pinned);
  const togglePinned = usePinnedModelsStore((s) => s.togglePinned);
  const pinned = pinnedKeys.includes(pinKey(repoId, quant));
  const deviceType = usePlatformStore((s) => s.deviceType);
  const revealLabel =
    deviceType === "mac"
      ? "Reveal in Finder"
      : deviceType === "windows"
        ? "Reveal in File Explorer"
        : "Reveal in File Manager";
  const handleCopyPath = useCallback(async () => {
    try {
      const { path } = await getCachedModelPath(repoId, quant);
      if (await copyToClipboard(path)) {
        toast.success("Copied path");
      } else {
        toast.error("Failed to copy");
      }
    } catch (err) {
      toast.error(
        err instanceof Error ? err.message : "Failed to resolve model path",
      );
    }
  }, [repoId, quant]);
  const handleCopyId = useCallback(async () => {
    const id = quant ? `${repoId}:${quant}` : repoId;
    if (await copyToClipboard(id)) {
      toast.success("Copied identifier");
    } else {
      toast.error("Failed to copy");
    }
  }, [repoId, quant]);

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild={true}>
        <button
          type="button"
          onClick={(e) => e.stopPropagation()}
          aria-label={`More options for ${label}`}
          className={cn(
            "inline-flex size-6 shrink-0 cursor-pointer items-center justify-center rounded-full",
            "text-muted-foreground transition-colors hover:bg-muted hover:text-foreground",
            "data-[state=open]:bg-muted data-[state=open]:text-foreground",
            buttonClassName,
          )}
        >
          <HugeiconsIcon
            icon={MoreVerticalIcon}
            strokeWidth={1.75}
            className={cn("size-3.5", iconClassName)}
          />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        side="bottom"
        align="end"
        sideOffset={2}
        className="unsloth-plus-menu menu-flat-destructive w-48"
      >
        {showPin && downloaded && (
          <DropdownMenuItem
            onSelect={(e) => {
              e.stopPropagation();
              togglePinned(repoId, quant);
            }}
          >
            <HugeiconsIcon
              icon={pinned ? PinOffIcon : PinIcon}
              strokeWidth={1.75}
              className="size-icon"
            />
            <span>{pinned ? "Unpin" : "Pin to top"}</span>
          </DropdownMenuItem>
        )}
        {downloaded && (
          <DropdownMenuItem
            onSelect={(e) => {
              e.stopPropagation();
              revealCachedModel(repoId, quant).catch((err) => {
                toast.error(
                  err instanceof Error
                    ? err.message
                    : "Failed to open file manager",
                );
              });
            }}
          >
            <HugeiconsIcon
              icon={Folder01Icon}
              strokeWidth={1.75}
              className="size-icon"
            />
            <span>{revealLabel}</span>
          </DropdownMenuItem>
        )}
        <DropdownMenuItem
          onSelect={(e) => {
            e.stopPropagation();
            void handleCopyId();
          }}
        >
          <HugeiconsIcon
            icon={Copy01Icon}
            strokeWidth={1.75}
            className="size-icon"
          />
          <span>Copy identifier</span>
        </DropdownMenuItem>
        {downloaded && (
          <DropdownMenuItem
            onSelect={(e) => {
              e.stopPropagation();
              void handleCopyPath();
            }}
          >
            <HugeiconsIcon
              icon={Copy01Icon}
              strokeWidth={1.75}
              className="size-icon"
            />
            <span>Copy path</span>
          </DropdownMenuItem>
        )}
        {canDelete && (
          <>
            <DropdownMenuSeparator />
            <DropdownMenuItem
              variant="destructive"
              onSelect={(e) => {
                e.stopPropagation();
                onDelete(quant);
              }}
            >
              <HugeiconsIcon
                icon={Delete02Icon}
                strokeWidth={1.75}
                className="size-icon"
              />
              <span>Delete</span>
            </DropdownMenuItem>
          </>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

const GgufVariantMenuRow = memo(function GgufVariantMenuRow({
  repoId,
  item,
  selected,
  loaded,
  liveActive,
  showFitInfo,
  onSelect,
  onDelete,
}: {
  repoId: string;
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
            <TooltipTrigger asChild={true}>
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
        <span className={cn(CHIP_BASE, CHIP_DEFAULT)}>
          {item.downloadSizeLabel}
        </span>
        {/* Options only apply to files on disk; placeholder keeps the size
            chips column-aligned across rows. */}
        {item.downloaded || item.partial ? (
          <QuantOptionsMenu
            repoId={repoId}
            quant={item.quant}
            label={item.label}
            downloaded={Boolean(item.downloaded)}
            canDelete={canDelete}
            onDelete={(q) => q && onDelete(q)}
          />
        ) : (
          <span aria-hidden={true} className="size-6 shrink-0" />
        )}
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
  onEject,
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
  /** Accepted for API parity; the run bar ejects instead of opening chat. */
  onUseInChat?: () => void;
  onEject?: () => void;
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
  const liveVariantStates = useDownloadManagerStore(
    selectLiveGgufVariantStates,
  );
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
  }, [variants, progress?.variant, progress?.expectedBytes, setExpectedBytes]);

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
    successMessage: () =>
      `Deleted ${repoId} ${deleteTargetLabel ?? deleteTarget}`,
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
        loading={true}
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
          partial={true}
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
          <PopoverTrigger asChild={true}>
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
              <span className="flex min-w-0 flex-1 items-center gap-2 overflow-hidden text-[0.75rem] text-muted-foreground">
                {selected ? (
                  <QuantBadge
                    quant={selectedLabel ?? selected.quant}
                    fit={selectedFit ?? "oom"}
                    showFit={showFitInfo}
                    active={Boolean(selectedIsActive)}
                  />
                ) : (
                  <span className="text-[0.78125rem] text-muted-foreground">
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
                    <TooltipTrigger asChild={true}>
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
                    repoId={repoId}
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

        {/* TODO: inference settings gear hidden for now, work on it in a future PR. */}
        {/* Options only resolve managed HF-cache repos, so skip local paths;
            they also only apply to quants actually on disk. */}
        {selected &&
          Boolean(selected.downloaded || selected.partial) &&
          !/^([/\\~.]|[A-Za-z]:)/.test(repoId) && (
            <QuantOptionsMenu
              repoId={repoId}
              quant={selected.quant}
              label={`${repoId} ${selectedLabel}`}
              downloaded={Boolean(selected.downloaded)}
              canDelete={
                Boolean(selected.downloaded || selected.partial) &&
                !selectedIsActive &&
                !downloadingThisVariant &&
                !isLoadingThisModel
              }
              onDelete={(q) => q && handleDeleteVariant(q)}
              showPin={false}
              buttonClassName="ml-0.5 size-7"
              iconClassName="size-4"
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
              onEject?.();
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
            isGgufRunCta ? "hub-run-action-btn w-24" : "hub-action-btn w-24",
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
              <HugeiconsIcon icon={RemoveCircleIcon} strokeWidth={1.75} />
              Eject
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
          className="self-start px-1 text-[0.6875rem] text-status-warning underline-offset-2 transition-colors hover:underline"
        >
          Couldn't refresh quantizations. Retry
        </button>
      )}
    </div>
  );
}
