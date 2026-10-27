// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  readLastLocalModelLoad,
  type useChatModelRuntime,
  useChatRuntimeStore,
} from "@/features/chat";
import { isPrecisionFolderName } from "@/features/hub/lib/dataset-size";
import {
  isNativeFileLabel,
  isStandaloneGgufPath,
} from "@/features/hub/lib/model-identity";
import { isNpuModelId } from "@/features/npu";
import {
  currentRuntimePerModelConfig,
  resolveResidentInitialConfig,
} from "@/features/model-picker";
import { useCallback, useEffect, useState } from "react";
import {
  type LastLoadRecord,
  RELOAD_MISSING_HISTORY_MESSAGE,
  reloadLastModel,
} from "./reload-last-model";
import {
  type ResidentRunTarget,
  residentHasRunSettings,
} from "./slots-settings-request";
import { monitorSelection } from "./use-resident-run-target";

type Runtime = ReturnType<typeof useChatModelRuntime>;

export type ReloadableModel = {
  name: string;
  checkpoint: string;
  quant: string | null;
  active: boolean;
  isMlx?: boolean;
};

function errorMessage(err: unknown, fallback: string): string | null {
  const runtime = useChatRuntimeStore.getState();
  // A load that another pick replaced is not a failure to report.
  if (runtime.modelLoading || runtime.loadingModelPick) {
    return null;
  }
  return err instanceof Error ? err.message : fallback;
}

type ReloadSelection = LastLoadRecord & {
  loadId?: string;
  nativePathToken?: string;
  nativePathExpiresAtMs?: number | null;
};

/** A file-picked GGUF is served under a display label; only the resident's file grant can load it. */
function grantedTarget(resident: ResidentRunTarget): ReloadSelection {
  return {
    id: resident.checkpoint,
    kind: resident.isGguf ? "gguf" : "model",
    ggufVariant: resident.ggufVariant,
    loadId: resident.loadId ?? undefined,
    nativePathToken: resident.nativePathToken ?? undefined,
    nativePathExpiresAtMs: resident.nativePathExpiresAtMs,
  };
}

function servedTarget(model: ReloadableModel): LastLoadRecord {
  if (isStandaloneGgufPath(model.checkpoint)) {
    return { id: model.checkpoint, kind: "gguf", ggufVariant: null };
  }
  // The checkpoint, not a kept slot's public alias; only the synthetic primary row's is its label.
  const { id, ggufVariant } = monitorSelection(model.name);
  const variant = model.quant ?? ggufVariant;
  const target = model.checkpoint === model.name ? id : model.checkpoint;
  // MLX reports its precision folder as the quant but loads as a model. Status says MLX only for the
  // primary; a kept row's folder is told by name, and names both formats use (bf16) by a GGUF repo.
  const mlxFolder =
    model.isMlx ||
    (variant != null && isPrecisionFolderName(variant) && !/gguf/i.test(target));
  return {
    id: target,
    kind: variant != null && !mlxFolder ? "gguf" : "model",
    ggufVariant: variant,
  };
}

export function useApiModelReload({
  selectModel,
  loadNpuModel,
  readModelLoadEpochs,
  fileGrant,
  onSettled,
  residencyKey,
}: {
  selectModel: Runtime["selectModel"];
  loadNpuModel: Runtime["loadNpuModel"];
  readModelLoadEpochs: Runtime["readModelLoadEpochs"];
  fileGrant: ResidentRunTarget | null;
  onSettled: () => void;
  residencyKey: string;
}) {
  // undefined while the record is read; null when there is none.
  const [lastLoadLabel, setLastLoadLabel] = useState<string | null | undefined>(
    undefined,
  );
  const [reloadingKey, setReloadingKey] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refreshLastLoad = useCallback(() => {
    readLastLocalModelLoad()
      .then((last) =>
        setLastLoadLabel(
          last
            ? last.kind === "gguf" && last.ggufVariant
              ? `${last.id} · ${last.ggufVariant}`
              : last.id
            : null,
        ),
      )
      .catch(() => setLastLoadLabel(null));
  }, []);

  // biome-ignore lint/correctness/useExhaustiveDependencies: residencyKey is the trigger, not an input
  useEffect(() => {
    refreshLastLoad();
  }, [refreshLastLoad, residencyKey]);

  const load = useCallback(
    async (target: ReloadSelection) => {
      const resolved = resolveResidentInitialConfig(
        target.id,
        target.ggufVariant,
      );
      const config = resolved.remembered ? resolved.config : null;
      // As the picker does: /validate cannot resolve an NPU id, so these skip selectModel.
      if (isNpuModelId(target.id)) {
        await loadNpuModel(target.id, {
          forceReload: true,
          config: config ?? undefined,
        });
        return;
      }
      await selectModel({
        id: target.id,
        isGguf: target.kind === "gguf",
        ggufVariant: target.ggufVariant ?? undefined,
        loadId: target.loadId,
        nativePathToken: target.nativePathToken,
        nativePathExpiresAtMs: target.nativePathExpiresAtMs,
        config: config ?? undefined,
        keepSpeculative: config != null,
        forceReload: true,
        isDownloaded: true,
        previousConfig: currentRuntimePerModelConfig({
          includeMaxSeqLength: true,
        }),
        throwOnError: true,
      });
    },
    [selectModel, loadNpuModel],
  );

  const run = useCallback(
    async (key: string, work: () => Promise<void>) => {
      setReloadingKey(key);
      setError(null);
      try {
        await work();
        refreshLastLoad();
        onSettled();
      } catch (err: unknown) {
        setError(errorMessage(err, "Failed to reload model"));
      } finally {
        setReloadingKey(null);
      }
    },
    [onSettled, refreshLastLoad],
  );

  const rowTarget = useCallback(
    (model: ReloadableModel): ReloadSelection =>
      model.active && fileGrant
        ? grantedTarget(fileGrant)
        : servedTarget(model),
    [fileGrant],
  );

  // A file-picked GGUF is listed by its bare file name; without a live grant nothing here can load it.
  const canReload = useCallback(
    (model: ReloadableModel): boolean =>
      model.active && fileGrant
        ? residentHasRunSettings(fileGrant)
        : !isNativeFileLabel(model.checkpoint),
    [fileGrant],
  );

  const reloadServed = useCallback(
    (model: ReloadableModel) =>
      run(model.checkpoint, () => load(rowTarget(model))),
    [run, load, rowTarget],
  );

  // Kept models first and the primary last, so the primary is still the most recent load.
  const reloadAll = useCallback(
    (models: ReloadableModel[]) =>
      run("*", async () => {
        if (models.length === 0) {
          await reloadLastModel({
            readLastLoad: () => readLastLocalModelLoad(),
            resolveConfig: (id, variant) => {
              const resolved = resolveResidentInitialConfig(id, variant);
              return resolved.remembered ? { config: resolved.config } : null;
            },
            loadTarget: load,
          });
          return;
        }
        for (const model of models
          .filter(canReload)
          .sort((a, b) => Number(a.active) - Number(b.active))) {
          const before = readModelLoadEpochs();
          await load(rowTarget(model));
          // selectModel returns quietly on a Stop, a declined prompt or a newer pick; each load takes one intent.
          const after = readModelLoadEpochs();
          if (
            after.cancels !== before.cancels ||
            after.intent > before.intent + 1
          ) {
            return;
          }
        }
      }),
    [run, load, readModelLoadEpochs, rowTarget, canReload],
  );

  const reloadAllTitle = (served: ReloadableModel[]): string => {
    if (served.length > 1) return `Reload all ${served.length} models`;
    if (served.length === 1) return `Reload ${served[0].name}`;
    if (lastLoadLabel === undefined)
      return "Checking previously loaded model...";
    if (!lastLoadLabel) return RELOAD_MISSING_HISTORY_MESSAGE;
    return `Reload ${lastLoadLabel} (no model loaded)`;
  };

  return {
    lastLoadLabel,
    refreshLastLoad,
    reloadingKey,
    error,
    reloadServed,
    reloadAll,
    canReload,
    reloadAllTitle,
  };
}
