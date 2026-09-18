// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Load + Reload controls for the API monitor page (issue #11189).
// Reuses the Chat model-loading stack instead of constructing a partial
// LoadModelRequest: ModelSelector for picking + useChatModelRuntime.selectModel
// for the load (context, quant variant, GPU placement, LoRA, trust approval,
// KV cache, speculative decoding, active-generation confirm, in-flight guard).
// Reload resolves its target from the durable last-local-load record, never
// from the bounded/clearable monitor ring buffer alone.

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
  resolveResidentInitialConfig,
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
}: {
  /** data.active_model from the monitor poll, for tooltips/disabled state. */
  activeModel: string | null | undefined;
  /** Refresh the monitor snapshot after a load/unload settles. */
  onSettled: () => void;
}): ReactElement {
  const { selectModel, refresh, ejectModel } = useChatModelRuntime();
  const activeGgufVariant = useChatRuntimeStore((s) => s.activeGgufVariant);
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

  // Populate the shared catalogs so the picker lists the same models as Chat.
  // refresh() fills models/loras + re-pins the active checkpoint; it never
  // clears an external-provider selection.
  useEffect(() => {
    refresh({ includeLoras: false });
    refreshLocalModels();
    refreshLastLoadLabel();
  }, [refresh, refreshLocalModels, refreshLastLoadLabel]);

  useEffect(() => {
    if (activeModel !== undefined) {
      refreshLastLoadLabel();
    }
  }, [activeModel, refreshLastLoadLabel]);

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
          source: meta?.source,
          isLora: meta?.isLora,
          ggufVariant: meta?.ggufVariant,
          isDownloaded: meta?.isDownloaded,
          expectedBytes: meta?.expectedBytes,
          isGguf: meta?.isGguf,
          isDiffusion: meta?.isDiffusion,
          config:
            meta?.config ??
            (remembered.remembered ? remembered.config : undefined),
          nativePathToken: meta?.nativePathToken,
          nativePathExpiresAtMs: meta?.nativePathExpiresAtMs,
          forceReload: meta?.forceReload,
        });
        refreshLastLoadLabel();
        onSettled();
      } catch (err: unknown) {
        setActionError(
          err instanceof Error ? err.message : "Failed to load model",
        );
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
        readSelectedCheckpoint: () =>
          useChatRuntimeStore.getState().params.checkpoint,
        isExternalSelection: (id) => isExternalModelId(id),
        resolveConfig: (id, variant) => {
          const resolved = resolveResidentInitialConfig(id, variant);
          return resolved.remembered ? { config: resolved.config } : null;
        },
        loadTarget: async (target) => {
          await selectModel({
            id: target.id,
            isGguf: target.kind === "gguf",
            ggufVariant: target.ggufVariant ?? undefined,
            config: (target.config as never) ?? undefined,
            forceReload: true,
            isDownloaded: true,
          });
        },
      });
      refreshLastLoadLabel();
      onSettled();
    } catch (err: unknown) {
      setActionError(
        err instanceof Error ? err.message : "Failed to reload model",
      );
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

  const effectiveModel = activeModel ?? "";
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
        onEject={() => {
          ejectModel().then(() => onSettled());
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
        title={activeModel ? reloadTitle : `${reloadTitle} (no model loaded)`}
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
