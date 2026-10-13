// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Loads through the Chat runtime's selectModel, so the API board gets the same context, quant,
// GPU placement, trust and running-chat guards as Chat.

import { Button } from "@/components/ui/button";
import {
  chatLocalModelOptions,
  isExternalModelId,
  readLastLocalModelLoad,
  useChatModelRuntime,
  useChatRuntimeStore,
  wantsDownloadManagerStaging,
} from "@/features/chat";
import {
  DOWNLOAD_KIND,
  downloadManager,
  publicModelId,
  useDeviceInventorySources,
  useRepoDownload,
} from "@/features/hub";
import {
  type LoraModelOption,
  type ModelOption,
  ModelSelector,
  type ModelSelectorChangeMeta,
  currentRuntimePerModelConfig,
  resolveResidentInitialConfig,
  splitQuantSuffix,
  useActiveModelConfig,
} from "@/features/model-picker";
import { isNpuModelId } from "@/features/npu";
import { cn } from "@/lib/utils";
import { Download01Icon, RefreshIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type Dispatch,
  type ReactElement,
  type SetStateAction,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";
import {
  RELOAD_MISSING_HISTORY_MESSAGE,
  reloadLastModel,
} from "../reload-last-model";

// A pick superseded by a newer one rejects too; the newer pick owns the outcome.
function loadErrorMessage(err: unknown, fallback: string): string | null {
  const runtime = useChatRuntimeStore.getState();
  if (runtime.modelLoading || runtime.loadingModelPick) {
    return null;
  }
  return err instanceof Error ? err.message : fallback;
}

// An uncached Hub pick downloads through the manager first, so the resident model keeps serving
// the API until the download completes and the pick loads.
async function startStagedDownload(
  id: string,
  meta: ModelSelectorChangeMeta | undefined,
): Promise<string | null> {
  const outcome = await downloadManager.requestStart({
    kind: DOWNLOAD_KIND.MODEL,
    repoId: id,
    variant: meta?.ggufVariant ?? null,
    expectedBytes: meta?.expectedBytes ?? 0,
    presentation: meta?.downloadPresentation,
    callerToast: {
      title: "Downloading model",
      description: "It'll load once the download finishes.",
    },
  });
  if (outcome === "conflict") {
    return "Resume this download from the Model hub.";
  }
  if (outcome === "busy") {
    return "A download for this model is already in progress.";
  }
  if (outcome === "error") {
    return "Failed to start the download.";
  }
  return null;
}

type SelectModelInput = Parameters<
  ReturnType<typeof useChatModelRuntime>["selectModel"]
>[0];

function localSelection(
  value: string,
  meta: ModelSelectorChangeMeta | undefined,
): SelectModelInput {
  const remembered = resolveResidentInitialConfig(
    value,
    meta?.ggufVariant ?? null,
  );
  const config =
    meta?.config ?? (remembered.remembered ? remembered.config : undefined);
  return {
    id: value,
    loadId: meta?.loadId,
    source: meta?.source,
    isLora: meta?.isLora,
    ggufVariant: meta?.ggufVariant,
    isDownloaded: meta?.isDownloaded,
    expectedBytes: meta?.expectedBytes,
    downloadPresentation: meta?.downloadPresentation,
    isGguf: meta?.isGguf,
    isVision: meta?.isVision,
    isDiffusion: meta?.isDiffusion,
    config,
    // As in Chat: a per-model config's speculative mode must not become the global preference.
    keepSpeculative: config !== undefined,
    nativePathToken: meta?.nativePathToken,
    nativePathExpiresAtMs: meta?.nativePathExpiresAtMs,
    forceReload: meta?.forceReload,
    previousConfig: currentRuntimePerModelConfig({ includeMaxSeqLength: true }),
    throwOnError: true,
  };
}

// The monitor reports a llama.cpp model as "<id>:<quant>"; Ollama ids carry their own ":<tag>".
function monitorSelection(activeModel: string | null | undefined): {
  id: string;
  ggufVariant: string | null;
} {
  if (!activeModel) {
    return { id: "", ggufVariant: null };
  }
  const split = activeModel.startsWith("ollama/")
    ? null
    : splitQuantSuffix(activeModel);
  return split
    ? { id: split[0], ggufVariant: split[1] }
    : { id: activeModel, ggufVariant: null };
}

type PendingDownload = {
  id: string;
  meta: ModelSelectorChangeMeta;
  originModel: string | null;
};

// Starts a staged pick's managed download and loads it on completion, as Chat does. The listener
// is bound before the start, and a newer pick cancels this one's start result.
function useLoadAfterDownload(
  pending: PendingDownload | null,
  setPending: Dispatch<SetStateAction<PendingDownload | null>>,
  activeModel: string | null,
  load: (id: string, meta: ModelSelectorChangeMeta) => void,
  onStartError: (message: string) => void,
): void {
  const settle = (variant: string | null, loadIt: boolean) => {
    if (!pending || (pending.meta.ggufVariant ?? null) !== (variant ?? null)) {
      return;
    }
    setPending(null);
    // An unload, reload or API switch since the pick means the user moved on.
    if (loadIt && pending.originModel === activeModel) {
      load(pending.id, { ...pending.meta, isDownloaded: true });
    }
  };
  useRepoDownload({
    kind: DOWNLOAD_KIND.MODEL,
    repoId: pending?.id ?? "__hub_autoload_idle__",
    activeVariant: pending?.meta.ggufVariant ?? null,
    onComplete: (variant) => settle(variant, true),
    onError: (variant) => settle(variant, false),
    onCancelled: (variant) => settle(variant, false),
  });
  useEffect(() => {
    if (!pending) {
      return;
    }
    let active = true;
    startStagedDownload(pending.id, pending.meta).then((error) => {
      if (active && error) {
        onStartError(error);
        setPending(null);
      }
    });
    return () => {
      active = false;
    };
  }, [pending, setPending, onStartError]);
}

function reloadTitle(
  lastLoadLabel: string | null | undefined,
  loaded: boolean,
): string {
  if (lastLoadLabel === undefined) {
    return "Checking previously loaded model...";
  }
  if (!lastLoadLabel) {
    return RELOAD_MISSING_HISTORY_MESSAGE;
  }
  return loaded
    ? `Reload ${lastLoadLabel}`
    : `Reload ${lastLoadLabel} (no model loaded)`;
}

function toModelOptions(
  models: {
    id: string;
    name: string;
    description?: string;
    isGguf?: boolean;
  }[],
): ModelOption[] {
  return models.map((model) => ({
    id: model.id,
    name: model.name,
    description: model.description,
    isGguf: model.isGguf,
  }));
}

export function ApiModelLoadControls({
  activeModel,
  onSettled,
  onUnloadActive,
  unloading,
}: {
  activeModel: string | null | undefined;
  onSettled: () => void;
  onUnloadActive: () => void;
  // The page's unload rereads the resident on its second pass, so a load started meanwhile would be unloaded.
  unloading: boolean;
}): ReactElement {
  const { selectModel, loadNpuModel, refresh, ejectModel } =
    useChatModelRuntime();
  const modelsFromStore = useChatRuntimeStore((s) => s.models);
  const lorasFromStore = useChatRuntimeStore((s) => s.loras);
  const modelLoading = useChatRuntimeStore((s) => s.modelLoading);
  const loadedContextLength = useChatRuntimeStore((s) => s.loadedContextLength);
  const { checkpoint: runtimeCheckpoint, config: runtimeConfig } =
    useActiveModelConfig();
  const localModelInventory = useDeviceInventorySources(["localModels"]);
  const refreshLocalModels = localModelInventory.refresh;

  const [selectorOpen, setSelectorOpen] = useState(false);
  const [reloading, setReloading] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [pendingDownload, setPendingDownload] =
    useState<PendingDownload | null>(null);
  const [lastLoadLabel, setLastLoadLabel] = useState<string | null | undefined>(
    undefined,
  );

  const refreshLastLoadLabel = useCallback(() => {
    readLastLocalModelLoad()
      .then((last) => {
        setLastLoadLabel(
          last
            ? last.kind === "gguf" && last.ggufVariant
              ? `${last.id} · ${last.ggufVariant}`
              : last.id
            : null,
        );
      })
      .catch(() => setLastLoadLabel(null));
  }, []);

  useEffect(() => {
    refresh();
    refreshLocalModels();
    refreshLastLoadLabel();
  }, [refresh, refreshLocalModels, refreshLastLoadLabel]);

  // An API auto-switch changes the resident model behind Chat's store; this page's own loads
  // reconcile themselves.
  useEffect(() => {
    if (activeModel === undefined) {
      return;
    }
    refreshLastLoadLabel();
    if (!useChatRuntimeStore.getState().modelLoading) {
      refresh({ includeLoras: false });
    }
  }, [activeModel, refresh, refreshLastLoadLabel]);

  const models = useMemo(
    () => toModelOptions(modelsFromStore),
    [modelsFromStore],
  );

  const loraModels = useMemo<LoraModelOption[]>(() => {
    const fromLoras = lorasFromStore.map((lora) => ({
      id: lora.id,
      name: lora.name,
      baseModel: lora.baseModel,
      updatedAt: lora.updatedAt,
      source: lora.source,
      exportType: lora.exportType,
    }));
    return [
      ...fromLoras,
      ...chatLocalModelOptions(localModelInventory.localModels.rows),
    ];
  }, [lorasFromStore, localModelInventory.localModels.rows]);

  const handlePick = useCallback(
    async (value: string, meta?: ModelSelectorChangeMeta) => {
      if (!value || unloading) {
        return;
      }
      setActionError(null);
      setSelectorOpen(false);
      if (meta && wantsDownloadManagerStaging({ id: value, ...meta })) {
        // A repeat pick of the download already pending keeps it rather than starting another.
        setPendingDownload((current) =>
          current?.id === value &&
          (current.meta.ggufVariant ?? null) === (meta.ggufVariant ?? null)
            ? current
            : { id: value, meta, originModel: activeModel ?? null },
        );
        return;
      }
      setPendingDownload(null);
      if (meta?.source === "external" || isExternalModelId(value)) {
        setActionError("External provider models are not served by the API.");
        onSettled();
        return;
      }
      if (isNpuModelId(value)) {
        await loadNpuModel(value, {
          forceReload: meta?.forceReload,
          config: meta?.config,
        });
        refreshLastLoadLabel();
        onSettled();
        return;
      }
      try {
        await selectModel(localSelection(value, meta));
        refreshLastLoadLabel();
        onSettled();
      } catch (err: unknown) {
        setActionError(loadErrorMessage(err, "Failed to load model"));
      }
    },
    [
      selectModel,
      loadNpuModel,
      onSettled,
      refreshLastLoadLabel,
      activeModel,
      unloading,
    ],
  );

  useLoadAfterDownload(
    pendingDownload,
    setPendingDownload,
    activeModel ?? null,
    (id, meta) => {
      handlePick(id, meta);
    },
    setActionError,
  );

  const handleReload = useCallback(async () => {
    setPendingDownload(null);
    setReloading(true);
    setActionError(null);
    try {
      await reloadLastModel({
        readLastLoad: () => readLastLocalModelLoad(),
        resolveConfig: (id, variant) => {
          const resolved = resolveResidentInitialConfig(id, variant);
          return resolved.remembered ? { config: resolved.config } : null;
        },
        loadTarget: async (target) => {
          await selectModel({
            id: target.id,
            isGguf: target.kind === "gguf",
            ggufVariant: target.ggufVariant ?? undefined,
            config: target.config ?? undefined,
            keepSpeculative: target.config != null,
            forceReload: true,
            isDownloaded: true,
            previousConfig: currentRuntimePerModelConfig({
              includeMaxSeqLength: true,
            }),
            throwOnError: true,
          });
        },
      });
      refreshLastLoadLabel();
      onSettled();
    } catch (err: unknown) {
      setActionError(loadErrorMessage(err, "Failed to reload model"));
    } finally {
      setReloading(false);
    }
  }, [selectModel, onSettled, refreshLastLoadLabel]);

  const handleEject = useCallback(
    (modelId?: string) => {
      setPendingDownload(null);
      if (modelId) {
        ejectModel(modelId).then(() => onSettled());
      } else {
        onUnloadActive();
      }
    },
    [ejectModel, onSettled, onUnloadActive],
  );

  const busy = modelLoading || unloading;
  const reloadDisabled = reloading || busy || lastLoadLabel == null;

  const { id: effectiveModel, ggufVariant: activeGgufVariant } =
    monitorSelection(activeModel);
  const isLoaded = Boolean(activeModel);
  // Chat's live settings describe the resident only when its checkpoint is the one the monitor reports.
  // The monitor reports a local GGUF by its public id (file stem), Chat's runtime by its path.
  const runtimeIsResident =
    isLoaded &&
    runtimeCheckpoint != null &&
    publicModelId(runtimeCheckpoint) === effectiveModel;

  return (
    <div className="flex flex-wrap items-center gap-2">
      <ModelSelector
        models={models}
        loraModels={loraModels}
        value={effectiveModel}
        loaded={isLoaded}
        activeGgufVariant={isLoaded ? activeGgufVariant : null}
        activeModelConfig={runtimeIsResident ? runtimeConfig : null}
        activeLoadedContextLength={
          runtimeIsResident ? loadedContextLength : null
        }
        onValueChange={(value, meta) => {
          handlePick(value, meta);
        }}
        // The page unload skips the runtime's load guard, so a replacement mid-load would land after it.
        onEject={busy ? undefined : handleEject}
        onFoldersChange={() => {
          refreshLocalModels();
        }}
        onModelsChange={() => {
          refresh();
          refreshLocalModels();
        }}
        deleteDisabled={modelLoading}
        variant="outline"
        size="sm"
        placeholder="Load model"
        open={selectorOpen && !unloading}
        onOpenChange={setSelectorOpen}
        className={cn("h-9 rounded-full")}
      />
      <Button
        type="button"
        variant="outline"
        size="sm"
        onClick={() => {
          handleReload();
        }}
        disabled={reloadDisabled}
        title={reloadTitle(lastLoadLabel, Boolean(activeModel))}
        className="h-9 gap-1.5 rounded-full"
      >
        <HugeiconsIcon
          icon={reloading ? RefreshIcon : Download01Icon}
          strokeWidth={1.75}
          className={cn("size-4", reloading && "animate-spin")}
        />
        {reloading ? "Reloading" : "Reload"}
      </Button>
      {actionError ? (
        <span
          role="alert"
          className="max-w-64 truncate text-ui-11 text-red-600 dark:text-red-400"
          title={actionError}
        >
          {actionError}
        </span>
      ) : null}
    </div>
  );
}
