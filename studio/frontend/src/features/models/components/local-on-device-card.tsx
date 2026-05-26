// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { TrainIcon } from "@/components/icons/train-icon";
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
  type BaseModelSource,
  type GgufVariantDetail,
  type LocalModelInfo,
  type ModelInventoryFormat,
  deleteCachedModel,
  listGgufVariants,
  useGgufVariantsCacheVersion,
} from "@/features/inventory";
import { formatBytes } from "@/lib/format";
import { ggufVariantsMatch } from "@/lib/model-identity";
import { cn } from "@/lib/utils";
import { useHfTokenStore } from "@/stores/hf-token-store";
import {
  Alert02Icon,
  ArrowDown01Icon,
  CubeIcon,
  Delete02Icon,
  PencilEdit02Icon,
  PlayIcon,
  Share05Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import { notifyInventoryEntryDeleted } from "../delete-notifications";
import { sortLocalGgufVariants } from "../lib/gguf-variant-sort";
import { DotTag } from "./dot-tag";
import { DeleteConfirmDialog } from "./download-card";
import { PathInfoButton } from "./path-info-button";
import { PerModelConfigNotice } from "./per-model-config-notice";

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
  onUseInChat: () => void;
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
          <TooltipTrigger asChild>
            <a
              href={`https://huggingface.co/${baseModelHubId}`}
              target="_blank"
              rel="noopener noreferrer"
              aria-label={`Open ${baseModelHubId} on Hugging Face`}
              className="inline-flex size-7 shrink-0 items-center justify-center rounded-[8px] text-muted-foreground transition-colors hover:bg-background hover:text-foreground"
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
  onUseInChat,
  onTrain,
  onChange,
}: LocalOnDeviceCardProps) {
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [variantOpen, setVariantOpen] = useState(false);
  const hfToken = useHfTokenStore((s) => s.token);
  const variantsVersion = useGgufVariantsCacheVersion(modelId);
  const localGgufPath = path.trim();
  const needsVariantSelection =
    isGguf && requiresVariant && !localGgufPath.toLowerCase().endsWith(".gguf");
  const variantKey = `${modelId}::${localGgufPath}::${variantsVersion}`;
  const [variantState, setVariantState] = useState<{
    key: string;
    variants: GgufVariantDetail[] | null;
    defaultVariant: string | null;
    loading: boolean;
    error: string | null;
  }>(() => ({
    key: variantKey,
    variants: null,
    defaultVariant: null,
    loading: needsVariantSelection,
    error: null,
  }));
  const [selectedVariantState, setSelectedVariantState] = useState<{
    key: string;
    quant: string | null;
  }>(() => ({
    key: variantKey,
    quant: null,
  }));

  const canDelete =
    source === "hf_cache" && !!repoId && !isActive && !isLoading;

  useEffect(() => {
    if (!needsVariantSelection) return;
    let cancelled = false;
    listGgufVariants(modelId, hfToken || undefined, {
      preferLocalCache: true,
      localPath: localGgufPath,
    })
      .then((response) => {
        if (cancelled) return;
        setVariantState({
          key: variantKey,
          variants: response.variants,
          defaultVariant: response.default_variant,
          loading: false,
          error: null,
        });
      })
      .catch((err) => {
        if (cancelled) return;
        setVariantState({
          key: variantKey,
          variants: null,
          defaultVariant: null,
          loading: false,
          error:
            err instanceof Error ? err.message : "Failed to load quantizations",
        });
      });
    return () => {
      cancelled = true;
    };
  }, [needsVariantSelection, variantKey, modelId, hfToken, localGgufPath]);

  const currentVariantState =
    variantState.key === variantKey
      ? variantState
      : {
          key: variantKey,
          variants: null,
          defaultVariant: null,
          loading: needsVariantSelection,
          error: null,
        };
  const variants = currentVariantState.variants;
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

  async function handleDelete() {
    if (!repoId) return;
    setDeleting(true);
    try {
      await deleteCachedModel(repoId, undefined, hfToken || undefined);
      notifyInventoryEntryDeleted({ kind: "model", id: repoId });
      toast.success(`Deleted ${repoId}`);
      setDeleteOpen(false);
      onChange?.();
    } catch (err) {
      toast.error("Failed to delete model", {
        description: err instanceof Error ? err.message : undefined,
      });
    } finally {
      setDeleting(false);
    }
  }

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
      <PerModelConfigNotice
        modelId={modelId}
        ggufVariant={needsVariantSelection ? selectedQuant : null}
      />
      <div className="download-card">
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
                            : (selectedQuant ?? "Select")}
                      </span>
                      {selectedVariant && (
                        <span className="shrink-0 font-sans text-[10px] text-muted-foreground tabular-nums">
                          {formatBytes(selectedVariant.size_bytes)}
                        </span>
                      )}
                      <HugeiconsIcon
                        icon={ArrowDown01Icon}
                        strokeWidth={1.5}
                        className="size-3 shrink-0"
                      />
                    </button>
                  </PopoverTrigger>
                  <PopoverContent
                    align="start"
                    side="bottom"
                    sideOffset={8}
                    avoidCollisions={false}
                    noAnimation
                    className="menu-instant menu-soft-surface w-[var(--radix-popover-trigger-width)] min-w-[220px] gap-0 overflow-hidden p-0 py-2 ring-0"
                  >
                    <div className="max-h-[280px] overflow-y-auto [scrollbar-width:thin]">
                      {sortedVariants?.map((variant) => {
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
                              {variant.quant}
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
              {canDelete && (
                <Tooltip>
                  <TooltipTrigger asChild={true}>
                    <button
                      type="button"
                      aria-label={`Delete ${repoId}`}
                      onClick={(e) => {
                        e.stopPropagation();
                        setDeleteOpen(true);
                      }}
                      className="inline-flex size-7 shrink-0 cursor-pointer items-center justify-center rounded-[8px] text-muted-foreground opacity-0 transition-[opacity,background-color,color] duration-150 hover:bg-rose-500/10 hover:text-rose-600 focus-visible:opacity-100 group-hover/dl:opacity-100 dark:hover:bg-rose-500/15 dark:hover:text-rose-400"
                    >
                      <HugeiconsIcon
                        icon={Delete02Icon}
                        strokeWidth={1.75}
                        className="size-4"
                      />
                    </button>
                  </TooltipTrigger>
                  <TooltipContent side="top" sideOffset={4}>
                    Delete from device
                  </TooltipContent>
                </Tooltip>
              )}
              <PathInfoButton
                path={path}
                title={sourceLabel}
                description="Where this model lives on disk."
              />
            </div>
          </div>
          {onTrain && (
            <div
              aria-hidden="true"
              className="ml-1 mr-0 h-5 w-px shrink-0 bg-foreground/[0.06] opacity-100 transition-opacity duration-150 group-hover/dl:opacity-0 dark:bg-white/[0.04]"
            />
          )}
          <div className="group/pair flex h-9 shrink-0 items-stretch gap-1.5">
            {onTrain && (
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
                  onUseInChat();
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
                  : "run-action-btn w-24",
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
                  <HugeiconsIcon icon={PencilEdit02Icon} strokeWidth={1.75} />
                  Chat
                </>
              ) : variantActionPending ? (
                <>
                  <Spinner />
                  Loading…
                </>
              ) : !canRun ? (
                <>
                  <HugeiconsIcon icon={Alert02Icon} strokeWidth={1.75} />
                  No run
                </>
              ) : (
                <>
                  <HugeiconsIcon icon={PlayIcon} strokeWidth={1.75} />
                  Run
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
        onConfirm={() => void handleDelete()}
        description={
          <>
            This will remove{" "}
            <span className="font-medium text-foreground">{repoId}</span> and
            its downloaded files from disk. You can re-download it later.
          </>
        }
      />
    </div>
  );
}
