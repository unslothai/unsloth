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
} from "@/features/chat";
import { useDeviceInventorySources } from "@/features/hub";
import {
  type LoraModelOption,
  type ModelOption,
  ModelSelector,
  type ModelSelectorChangeMeta,
  currentRuntimePerModelConfig,
  resolveResidentInitialConfig,
  splitQuantSuffix,
} from "@/features/model-picker";
import { cn } from "@/lib/utils";
import { Download01Icon, RefreshIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type ReactElement,
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
}: {
  activeModel: string | null | undefined;
  onSettled: () => void;
  onUnloadActive: () => void;
}): ReactElement {
  const { selectModel, refresh, ejectModel } = useChatModelRuntime();
  const modelsFromStore = useChatRuntimeStore((s) => s.models);
  const lorasFromStore = useChatRuntimeStore((s) => s.loras);
  const modelLoading = useChatRuntimeStore((s) => s.modelLoading);
  const localModelInventory = useDeviceInventorySources(["localModels"]);
  const refreshLocalModels = localModelInventory.refresh;

  const [selectorOpen, setSelectorOpen] = useState(false);
  const [reloading, setReloading] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
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
      if (!value) {
        return;
      }
      setActionError(null);
      setSelectorOpen(false);
      if (meta?.source === "external" || isExternalModelId(value)) {
        setActionError("External provider models are not served by the API.");
        onSettled();
        return;
      }
      try {
        const remembered = resolveResidentInitialConfig(
          value,
          meta?.ggufVariant ?? null,
        );
        await selectModel({
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
          config:
            meta?.config ??
            (remembered.remembered ? remembered.config : undefined),
          nativePathToken: meta?.nativePathToken,
          nativePathExpiresAtMs: meta?.nativePathExpiresAtMs,
          forceReload: meta?.forceReload,
          previousConfig: currentRuntimePerModelConfig({
            includeMaxSeqLength: true,
          }),
          throwOnError: true,
        });
        refreshLastLoadLabel();
        onSettled();
      } catch (err: unknown) {
        setActionError(loadErrorMessage(err, "Failed to load model"));
      }
    },
    [selectModel, onSettled, refreshLastLoadLabel],
  );

  const handleReload = useCallback(async () => {
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

  const reloadDisabled = reloading || modelLoading || lastLoadLabel == null;
  const reloadTitle =
    lastLoadLabel === undefined
      ? "Checking previously loaded model..."
      : lastLoadLabel
        ? `Reload ${lastLoadLabel}`
        : RELOAD_MISSING_HISTORY_MESSAGE;

  // The monitor reports a llama.cpp model as "<id>:<quant>"; Ollama ids carry their own ":<tag>".
  const quantSplit =
    activeModel && !activeModel.startsWith("ollama/")
      ? splitQuantSuffix(activeModel)
      : null;
  const effectiveModel = quantSplit?.[0] ?? activeModel ?? "";
  const activeGgufVariant = quantSplit?.[1] ?? null;
  const isLoaded = Boolean(activeModel);

  return (
    <div className="flex flex-wrap items-center gap-2">
      <ModelSelector
        models={models}
        loraModels={loraModels}
        value={effectiveModel}
        loaded={isLoaded}
        activeGgufVariant={isLoaded ? activeGgufVariant : null}
        onValueChange={(value, meta) => {
          handlePick(value, meta);
        }}
        onEject={(modelId) => {
          if (modelId) {
            ejectModel(modelId).then(() => onSettled());
          } else {
            onUnloadActive();
          }
        }}
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
        open={selectorOpen}
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
        title={
          activeModel || !lastLoadLabel
            ? reloadTitle
            : `${reloadTitle} (no model loaded)`
        }
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
