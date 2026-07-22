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
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { cn } from "@/lib/utils";
import {
  Alert02Icon,
  CubeIcon,
  PlayIcon,
  RemoveCircleIcon,
  Share05Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useMemo, useState } from "react";
import { TrainIcon } from "../components/train-icon";
import {
  downloadManager,
  jobKeyOf,
  selectActiveJob,
  useDownloadManagerStore,
} from "../download-manager";
import { useOnlineStatus } from "../hooks/use-online-status";
import {
  type BaseModelSource,
  type LocalModelInfo,
  type ModelInventoryFormat,
  deleteCachedModel,
} from "../inventory";
import { formatBytes } from "../lib/format";
import {
  ggufVariantDisplayLabel,
  sortLocalGgufVariants,
} from "../lib/gguf-variant-sort";
import {
  HUB_GGUF_RUN_ACTIONS_VISIBLE,
  HUB_NON_GGUF_RUN_ACTIONS_VISIBLE,
  HUB_POST_DOWNLOAD_ACTIONS_VISIBLE,
} from "../lib/hub-feature-flags";
import { ggufVariantsMatch } from "../lib/model-identity";
import { confirmExternalLink } from "../stores/external-link-confirm";
import { useHfTokenStore } from "../stores/hf-token-store";
import { DotTag } from "./dot-tag";
import {
  CardDeleteButton,
  CardUpdateButton,
  DeleteConfirmDialog,
  UpdateConfirmDialog,
} from "./download-card";
import { PathInfoButton } from "./path-info-button";
import { TransportConflictDialog } from "./transport-conflict-dialog";
import { useCardDelete } from "./use-card-delete";
import { useGgufVariantFetchState } from "./use-gguf-variant-fetch-state";

type LocalLoadOptions = {
  ggufVariant?: string;
  expectedBytes?: number;
};

interface LocalOnDeviceCardProps {
  modelId: string;
  repoId: string | null;
  sourceLabel: string;
  source: LocalModelInfo["source"];
  path: string;
  isGguf: boolean;
  requiresVariant?: boolean;
  modelFormat: ModelInventoryFormat | null;
  baseModel?: string | null;
  baseModelSource?: BaseModelSource | null;
  baseModelHubId?: string | null;
  baseModelSummary?: string | null;
  adapterType?: string | null;
  trainingMethod?: string | null;
  canRun?: boolean;
  isActive: boolean;
  activeGgufVariant?: string | null;
  isLoading: boolean;
  loadingPhase?: "downloading" | "starting";
  gpuGb?: number;
  systemRamGb?: number;
  unsupportedReason?: string | null;
  onLoad: (opts?: LocalLoadOptions) => void;
  /** Accepted for API parity; the run bar ejects instead of opening chat. */
  onUseInChat: () => void;
  onEject?: () => void;
  onTrain?: () => void;
  onChange?: () => void;
}

function formatAdapterLabel(
  adapterType?: string | null,
  trainingMethod?: string | null,
): string {
  const method = trainingMethod?.trim().toLowerCase();
  if (method === "qlora") return "QLoRA adapter";
  if (method === "lora") return "LoRA adapter";
  const type = adapterType?.trim();
  return type ? `${type.toUpperCase()} adapter` : "Adapter";
}

function baseModelSourceLabel(source?: BaseModelSource | null): string {
  if (source === "huggingface") return "Hugging Face base model";
  if (source === "local") return "Local base model";
  return "Base model";
}

function BaseModelReference({
  baseModel,
  baseModelSource,
  baseModelHubId,
  baseModelSummary,
}: {
  baseModel: string;
  baseModelSource?: BaseModelSource | null;
  baseModelHubId?: string | null;
  baseModelSummary?: string | null;
}) {
  const canOpenHub = baseModelSource === "huggingface" && !!baseModelHubId;
  return (
    <div className="flex min-w-0 items-center gap-2 rounded-[10px] border border-border/55 bg-muted/35 px-3 py-2">
      <div className="flex size-7 shrink-0 items-center justify-center rounded-[8px] bg-background/70 text-muted-foreground">
        <HugeiconsIcon
          icon={CubeIcon}
          strokeWidth={1.75}
          className="size-3.5"
        />
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex min-w-0 items-center gap-1.5">
          <span className="shrink-0 text-[11px] font-medium text-muted-foreground">
            {baseModelSourceLabel(baseModelSource)}
          </span>
          <span className="truncate text-[12px] font-medium text-foreground">
            {baseModel}
          </span>
        </div>
        {baseModelSummary && (
          <p className="mt-0.5 truncate text-[11px] text-muted-foreground">
            {baseModelSummary}
          </p>
        )}
      </div>
      {canOpenHub && (
        <Tooltip>
          <TooltipTrigger asChild={true}>
            <a
              href={`https://huggingface.co/${baseModelHubId}`}
              target="_blank"
              rel="noopener noreferrer"
              aria-label={`Open ${baseModelHubId} on Hugging Face`}
              className="inline-flex size-7 shrink-0 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-background hover:text-foreground"
              onClick={(event) => {
                event.stopPropagation();
                if (
                  confirmExternalLink(
                    `https://huggingface.co/${baseModelHubId}`,
                  )
                ) {
                  event.preventDefault();
                }
              }}
            >
              <HugeiconsIcon
                icon={Share05Icon}
                strokeWidth={1.75}
                className="size-3.5"
              />
            </a>
          </TooltipTrigger>
          <TooltipContent side="top" sideOffset={4}>
            Open base model
          </TooltipContent>
        </Tooltip>
      )}
    </div>
  );
}

export function LocalOnDeviceCard({
  modelId,
  repoId,
  sourceLabel,
  source,
  path,
  isGguf,
  requiresVariant = false,
  modelFormat,
  baseModel,
  baseModelSource,
  baseModelHubId,
  baseModelSummary,
  adapterType,
  trainingMethod,
  canRun = true,
  isActive,
  activeGgufVariant = null,
  isLoading,
  loadingPhase,
  gpuGb,
  systemRamGb,
  unsupportedReason,
  onLoad,
  onEject,
  onTrain,
  onChange,
}: LocalOnDeviceCardProps) {
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [updateOpen, setUpdateOpen] = useState(false);
  const [variantOpen, setVariantOpen] = useState(false);
  const [updateConflictKey, setUpdateConflictKey] = useState<string | null>(
    null,
  );
  const updateTransportConflict = useDownloadManagerStore((state) =>
    updateConflictKey
      ? (state.conflicts[updateConflictKey]?.info ?? null)
      : null,
  );
  const cancelUpdateConflict = useCallback(() => {
    if (updateConflictKey) downloadManager.cancelConflict(updateConflictKey);
    setUpdateConflictKey(null);
  }, [updateConflictKey]);
  const resumeUpdateConflict = useCallback(() => {
    if (!updateConflictKey) return;
    downloadManager.resumeConflict(updateConflictKey);
    setUpdateConflictKey(null);
  }, [updateConflictKey]);
  const restartUpdateConflict = useCallback(() => {
    if (!updateConflictKey) return;
    downloadManager.restartConflict(updateConflictKey);
    setUpdateConflictKey(null);
  }, [updateConflictKey]);
  const hfToken = useHfTokenStore((s) => s.token);
  // Update availability is derived from the GGUF variant metadata; offline rows
  // keep the button hidden because there is no remote revision to fetch.
  const online = useOnlineStatus();
  const { deleting, runDelete } = useCardDelete({
    action: async () => {
      if (!repoId) return;
      // Delete is only offered for hf_cache rows (see canDelete), so `path` is
      // the cache snapshot path: pass it so the delete targets the cache this
      // card shows instead of falling back to the active cache.
      await deleteCachedModel(repoId, undefined, hfToken || undefined, path);
    },
    resourceName: "model",
    successMessage: () => `Deleted ${repoId}`,
    onSuccess: () => {
      setDeleteOpen(false);
      onChange?.();
    },
  });
  const localGgufPath = path.trim();
  const needsVariantSelection =
    isGguf && requiresVariant && !localGgufPath.toLowerCase().endsWith(".gguf");
  const currentVariantState = useGgufVariantFetchState({
    repoId: modelId,
    hfToken,
    preferLocalCache: true,
    localPath: localGgufPath,
    enabled: needsVariantSelection,
    errorFallback: "Failed to load quantizations",
  });
  const remoteVariantState = useGgufVariantFetchState({
    repoId: repoId ?? modelId,
    hfToken,
    enabled:
      online && source === "hf_cache" && needsVariantSelection && !!repoId,
    errorFallback: "Failed to check for updates",
  });
  const variantKey = currentVariantState.key;
  const [selectedVariantState, setSelectedVariantState] = useState<{
    key: string;
    quant: string | null;
  }>(() => ({
    key: variantKey,
    quant: null,
  }));

  const canDelete =
    source === "hf_cache" && !!repoId && !isActive && !isLoading;
  const variants = useMemo(() => {
    const localVariants = currentVariantState.variants;
    const remoteVariants = remoteVariantState.variants;
    if (!localVariants || !remoteVariants) return localVariants;
    return localVariants.map((variant) => {
      const remoteVariant = remoteVariants.find((remote) =>
        ggufVariantsMatch(remote.quant, variant.quant),
      );
      if (!remoteVariant) return variant;
      return {
        ...variant,
        download_size_bytes:
          remoteVariant.download_size_bytes || variant.download_size_bytes,
        update_available: remoteVariant.update_available === true,
      };
    });
  }, [currentVariantState.variants, remoteVariantState.variants]);
  const sortedVariants = useMemo(
    () =>
      variants
        ? sortLocalGgufVariants(variants, {
            defaultVariant: currentVariantState.defaultVariant,
            activeGgufVariant: isActive ? activeGgufVariant : null,
            gpuGb,
            systemRamGb,
          })
        : null,
    [
      variants,
      currentVariantState.defaultVariant,
      isActive,
      activeGgufVariant,
      gpuGb,
      systemRamGb,
    ],
  );
  const selectedVariantOverride =
    selectedVariantState.key === variantKey ? selectedVariantState.quant : null;
  const selectedQuant =
    selectedVariantOverride &&
    sortedVariants?.some((variant) =>
      ggufVariantsMatch(variant.quant, selectedVariantOverride),
    )
      ? selectedVariantOverride
      : (sortedVariants?.find(
          (variant) =>
            isActive && ggufVariantsMatch(variant.quant, activeGgufVariant),
        )?.quant ??
        sortedVariants?.find((variant) =>
          ggufVariantsMatch(variant.quant, currentVariantState.defaultVariant),
        )?.quant ??
        sortedVariants?.[0]?.quant ??
        null);
  const selectedVariant =
    sortedVariants?.find((variant) =>
      ggufVariantsMatch(variant.quant, selectedQuant),
    ) ?? null;
  // True while a managed download/update for this repo+variant is in flight.
  const updateJobActive = useDownloadManagerStore((s) =>
    repoId
      ? Boolean(
          selectActiveJob(
            s,
            "model",
            repoId,
            needsVariantSelection ? selectedQuant : null,
          ),
        )
      : false,
  );
  const updateTargetVariant = needsVariantSelection ? selectedQuant : null;
  const updateExpectedBytes =
    selectedVariant?.download_size_bytes ?? selectedVariant?.size_bytes ?? 0;
  const updateAvailable =
    needsVariantSelection &&
    selectedVariant?.downloaded === true &&
    selectedVariant.update_available === true;
  const canUpdate =
    online &&
    source === "hf_cache" &&
    !!repoId &&
    !isActive &&
    !isLoading &&
    !updateJobActive &&
    updateAvailable;
  // Update runs as a MANAGED download (same path as a normal download) so it
  // shows in the Downloads panel with manifest-based progress and a working
  // Cancel. The worker re-resolves `main` and pulls changed blobs while the old
  // cached copy stays runnable until the new revision verifies.
  const handleConfirmUpdate = () => {
    if (!repoId || !updateTargetVariant) return;
    setUpdateOpen(false);
    void downloadManager
      .requestStart({
        kind: "model",
        repoId,
        variant: updateTargetVariant,
        expectedBytes: updateExpectedBytes,
      })
      .then((outcome) => {
        if (outcome === "conflict") {
          setUpdateConflictKey(jobKeyOf("model", repoId, updateTargetVariant));
        }
        void currentVariantState.refresh();
        void remoteVariantState.refresh();
      });
  };
  const selectedVariantIsActive =
    needsVariantSelection && selectedQuant
      ? isActive && ggufVariantsMatch(activeGgufVariant, selectedQuant)
      : isActive;
  const variantUnavailable =
    needsVariantSelection &&
    (currentVariantState.loading ||
      currentVariantState.error !== null ||
      selectedVariant === null);
  const variantActionPending =
    needsVariantSelection && currentVariantState.loading;

  const formatLabel =
    modelFormat === "gguf"
      ? "GGUF"
      : modelFormat === "adapter"
        ? formatAdapterLabel(adapterType, trainingMethod)
        : modelFormat === "checkpoint"
          ? "Checkpoint"
          : modelFormat === "safetensors"
            ? "Safetensors"
            : "Model";
  const formatTone =
    modelFormat === "adapter" ? "adapter" : isGguf ? "gguf" : "checkpoint";
  const showOldCacheHint = source === "hf_cache" && !!unsupportedReason;
  const runActionsVisible = isGguf
    ? HUB_GGUF_RUN_ACTIONS_VISIBLE
    : HUB_NON_GGUF_RUN_ACTIONS_VISIBLE;

  return (
    <div className="flex w-full flex-col gap-2">
      {showOldCacheHint && (
        <div className="flex items-start gap-2 rounded-[12px] border border-amber-500/20 bg-amber-500/8 px-3 py-2 text-[12px] leading-5 text-amber-700 dark:text-amber-300">
          <HugeiconsIcon
            icon={Alert02Icon}
            strokeWidth={1.75}
            className="mt-[1px] size-3.5 shrink-0"
          />
          <span>
            This looks like an older Hub cache. It may use a format that Unsloth
            no longer loads (missing config or unsupported quantization). You
            can still keep it on disk, or delete it to free space.
          </span>
        </div>
      )}
      <div className="hub-download-card">
        <div className="group/dl flex items-center">
          <div className="relative flex h-9 min-w-0 flex-1 items-center pl-3 pr-2">
            <span className="flex min-w-0 items-center gap-1.5 text-[12px] text-muted-foreground">
              <DotTag
                tone="success"
                label={selectedVariantIsActive ? "Loaded" : "On device"}
              />
              <DotTag tone={formatTone} label={formatLabel} />
              {needsVariantSelection && (
                <Popover open={variantOpen} onOpenChange={setVariantOpen}>
                  <PopoverTrigger asChild={true}>
                    <button
                      type="button"
                      disabled={currentVariantState.loading}
                      className="inline-flex h-6 max-w-[170px] shrink-0 cursor-pointer items-center gap-1.5 rounded-[8px] border border-format-gguf/35 px-2 font-mono text-[10.5px] leading-none text-format-gguf transition-colors hover:bg-format-gguf/8 disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      <span className="truncate">
                        {currentVariantState.loading
                          ? "Loading"
                          : currentVariantState.error
                            ? "Unavailable"
                            : selectedVariant
                              ? ggufVariantDisplayLabel(selectedVariant)
                              : "Select"}
                      </span>
                      {selectedVariant && (
                        <span className="shrink-0 font-sans text-[10px] text-muted-foreground tabular-nums">
                          {formatBytes(selectedVariant.size_bytes)}
                        </span>
                      )}
                      <HugeiconsIcon
                        icon={ChevronDownStandardIcon}
                        className="size-3 shrink-0"
                      />
                    </button>
                  </PopoverTrigger>
                  <PopoverContent
                    align="start"
                    side="bottom"
                    sideOffset={8}
                    avoidCollisions={false}
                    className="hub-menu-instant menu-soft-surface w-[var(--radix-popover-trigger-width)] min-w-[220px] gap-0 overflow-hidden p-0 py-2 ring-0"
                  >
                    <div className="max-h-[280px] overflow-y-auto [scrollbar-width:thin]">
                      {sortedVariants?.map((variant) => {
                        const label = ggufVariantDisplayLabel(variant);
                        const isSelected = ggufVariantsMatch(
                          variant.quant,
                          selectedQuant,
                        );
                        const isLoaded =
                          ggufVariantsMatch(variant.quant, activeGgufVariant) &&
                          isActive;
                        return (
                          <button
                            key={variant.filename}
                            type="button"
                            onClick={() => {
                              setSelectedVariantState({
                                key: variantKey,
                                quant: variant.quant,
                              });
                              setVariantOpen(false);
                            }}
                            className={cn(
                              "mx-2 flex w-[calc(100%-1rem)] min-w-0 cursor-pointer items-center gap-2 rounded-[10px] px-2.5 py-2 text-left transition-colors",
                              isSelected
                                ? "bg-foreground/[0.07] dark:bg-foreground/[0.12]"
                                : "hover:bg-foreground/[0.05] dark:hover:bg-foreground/[0.06]",
                            )}
                          >
                            <span className="min-w-0 flex-1 truncate font-mono text-[12px] text-format-gguf">
                              {label}
                            </span>
                            <span className="flex shrink-0 items-center gap-1.5">
                              {isLoaded && (
                                <DotTag tone="success" label="Loaded" />
                              )}
                              <span className="text-[10px] text-muted-foreground tabular-nums">
                                {formatBytes(variant.size_bytes)}
                              </span>
                            </span>
                          </button>
                        );
                      })}
                      {!currentVariantState.loading &&
                        !currentVariantState.error &&
                        sortedVariants?.length === 0 && (
                          <div className="px-4 py-2 text-xs text-muted-foreground">
                            No quantizations found.
                          </div>
                        )}
                      {currentVariantState.error && (
                        <div className="px-4 py-2 text-xs text-destructive">
                          {currentVariantState.error}
                        </div>
                      )}
                    </div>
                  </PopoverContent>
                </Popover>
              )}
              {source !== "hf_cache" && (
                <span className="truncate text-muted-foreground/85">
                  {sourceLabel}
                </span>
              )}
            </span>
            <div className="ml-auto flex items-center gap-0.5">
              {canUpdate && (
                <CardUpdateButton
                  label={`Update ${repoId}`}
                  emphasized={true}
                  onClick={() => setUpdateOpen(true)}
                />
              )}
              {canDelete && (
                <CardDeleteButton
                  label={`Delete ${repoId}`}
                  onClick={() => setDeleteOpen(true)}
                />
              )}
              <PathInfoButton path={path} />
            </div>
          </div>
          {onTrain && HUB_POST_DOWNLOAD_ACTIONS_VISIBLE && (
            <div
              aria-hidden="true"
              className="ml-1 mr-0 h-5 w-px shrink-0 bg-foreground/[0.06] opacity-100 transition-opacity duration-150 group-hover/dl:opacity-0 dark:bg-white/[0.04]"
            />
          )}
          <div
            className={cn(
              "group/pair flex h-9 shrink-0 items-stretch gap-1.5",
              !runActionsVisible && "hidden",
            )}
          >
            {onTrain && HUB_POST_DOWNLOAD_ACTIONS_VISIBLE && (
              <button
                type="button"
                onClick={() => onTrain()}
                className="hub-action-btn w-24"
              >
                <HugeiconsIcon icon={TrainIcon} strokeWidth={1.75} />
                Train
              </button>
            )}
            <button
              type="button"
              disabled={isLoading || variantUnavailable || !canRun}
              onClick={() => {
                if (!canRun) return;
                if (selectedVariantIsActive) {
                  onEject?.();
                  return;
                }
                if (needsVariantSelection) {
                  if (!selectedVariant) return;
                  onLoad({
                    ggufVariant: selectedVariant.quant,
                    expectedBytes: selectedVariant.size_bytes,
                  });
                  return;
                }
                onLoad();
              }}
              className={cn(
                isLoading ||
                  selectedVariantIsActive ||
                  variantUnavailable ||
                  !canRun
                  ? "hub-action-btn w-24"
                  : "hub-run-action-btn w-24",
                (isLoading || variantUnavailable || !canRun) && "opacity-70",
              )}
            >
              {isLoading ? (
                <>
                  <Spinner />
                  {loadingPhase === "downloading" ? "Preparing…" : "Loading…"}
                </>
              ) : selectedVariantIsActive ? (
                <>
                  <HugeiconsIcon icon={RemoveCircleIcon} strokeWidth={1.75} />
                  Eject
                </>
              ) : variantActionPending ? (
                <>
                  <Spinner />
                  Loading…
                </>
              ) : canRun ? (
                <>
                  <HugeiconsIcon icon={PlayIcon} strokeWidth={1.75} />
                  Run
                </>
              ) : (
                <>
                  <HugeiconsIcon icon={Alert02Icon} strokeWidth={1.75} />
                  No run
                </>
              )}
            </button>
          </div>
        </div>
      </div>
      {baseModel && (
        <BaseModelReference
          baseModel={baseModel}
          baseModelSource={baseModelSource}
          baseModelHubId={baseModelHubId}
          baseModelSummary={baseModelSummary}
        />
      )}
      <DeleteConfirmDialog
        open={deleteOpen}
        onOpenChange={(o) => {
          if (!o && !deleting) setDeleteOpen(false);
        }}
        title="Delete cached model?"
        deleting={deleting}
        onConfirm={() => void runDelete()}
        description={
          <>
            This will remove{" "}
            <span className="font-medium text-foreground">{repoId}</span> and
            its downloaded files from disk. You can re-download it later.
          </>
        }
      />
      <UpdateConfirmDialog
        open={updateOpen}
        onOpenChange={(o) => {
          if (!o) setUpdateOpen(false);
        }}
        title={`Update ${repoId}?`}
        updating={false}
        onConfirm={handleConfirmUpdate}
        description="Re-download the latest version of this model from Hugging Face. Progress shows in the Downloads panel."
      />
      <TransportConflictDialog
        conflict={updateTransportConflict}
        onCancel={cancelUpdateConflict}
        onKeepTransport={resumeUpdateConflict}
        onSwitchTransport={restartUpdateConflict}
      />
    </div>
  );
}
