// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Model selection for the Evals tab. It lives here, not in BenchmarkPage, because the
// picker sits in the page header next to the tabs — the same slot the config-sweep
// picker uses — and BenchmarksPage renders that header. Everything the evals run flow
// needs (pick, download watch, selected source/variant) hangs off this one object.

import { useChatModelRuntime, useChatRuntimeStore } from "@/features/chat";
import type {
  LoraModelOption,
  ModelOption,
  ModelSelectorChangeMeta,
} from "@/features/model-picker/components/model-selector/types";
import { type LocalModelInfo, listLocalModels } from "@/features/training";
import { useT } from "@/i18n";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { type ModelCheckpoints, fetchCheckpoints } from "./api/benchmark-api";

export interface EvalsModel {
  /** On-device models for the picker: local dirs, HF cache, custom folders. */
  models: ModelOption[];
  /** Fine-tuned checkpoints, the picker's LoRA section. */
  loraModels: LoraModelOption[];
  /** First failure from the two lists, if either could not load. */
  modelsError: string | null;
  loadingModels: boolean;

  selectedModel: string | null;
  selectedModelSource:
    | "hub"
    | "lora"
    | "exported"
    | "local"
    | "external"
    | null;
  selectedGgufVariant: string | null;
  selectedModelIsDownloaded: boolean | null;
  downloadingForBenchmark: boolean;

  /** Chat's runtime load in flight (truthy while a pick is loading), or null. */
  loadingModel: { id: string; displayName: string } | null;
  /** The checkpoint chat currently has resident, i.e. what "Loaded" means here. */
  residentModel: string | null;
  /** Loaded GGUF quant of the resident model. */
  activeGgufVariant: string | null;

  pick: (value: string, meta: ModelSelectorChangeMeta) => void;
  setDownloadingForBenchmark: (v: boolean) => void;
  setSelectedModelIsDownloaded: (v: boolean | null) => void;
  /** Swap the benchmark-download watcher; null drops it. */
  setDownloadWatch: (unsubscribe: (() => void) | null) => void;
  clearDownloadWatch: () => void;
}

export function useEvalsModel(): EvalsModel {
  const t = useT();

  const [trainingModels, setTrainingModels] = useState<ModelCheckpoints[]>([]);
  const [loadingCheckpoints, setLoadingCheckpoints] = useState(true);
  const [checkpointError, setCheckpointError] = useState<string | null>(null);

  const [localModels, setLocalModels] = useState<LocalModelInfo[]>([]);
  const [isLoadingLocalModels, setIsLoadingLocalModels] = useState(true);
  const [localModelsError, setLocalModelsError] = useState<string | null>(null);

  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [selectedModelSource, setSelectedModelSource] = useState<
    "hub" | "lora" | "exported" | "local" | "external" | null
  >(null);
  const [selectedGgufVariant, setSelectedGgufVariant] = useState<string | null>(
    null,
  );
  const [selectedModelIsDownloaded, setSelectedModelIsDownloaded] = useState<
    boolean | null
  >(null);
  const [downloadingForBenchmark, setDownloadingForBenchmark] = useState(false);

  const { selectModel, loadingModel, refresh } = useChatModelRuntime();
  const inferenceParams = useChatRuntimeStore((s) => s.params);
  const activeGgufVariant = useChatRuntimeStore((s) => s.activeGgufVariant);

  // Follow chat's runtime: a model loaded elsewhere (chat, Hub) becomes the pick, and
  // an unloaded runtime clears it.
  useEffect(() => {
    if (
      inferenceParams.checkpoint &&
      inferenceParams.checkpoint !== selectedModel
    ) {
      setSelectedModel(inferenceParams.checkpoint);
      setSelectedGgufVariant(activeGgufVariant);
    } else if (!inferenceParams.checkpoint && selectedModel) {
      setSelectedModel(null);
      setSelectedGgufVariant(null);
      setDownloadingForBenchmark(false);
    }
  }, [inferenceParams.checkpoint, activeGgufVariant, selectedModel]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    let cancelled = false;
    setLoadingCheckpoints(true);
    setCheckpointError(null);
    fetchCheckpoints()
      .then((data) => {
        if (!cancelled) {
          setTrainingModels(data.models);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setCheckpointError(
            err instanceof Error ? err.message : "Failed to load checkpoints",
          );
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoadingCheckpoints(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void listLocalModels(controller.signal)
      .then((models) => {
        if (controller.signal.aborted) {
          return;
        }
        setLocalModels(models);
      })
      .catch((error) => {
        if (controller.signal.aborted) {
          return;
        }
        setLocalModelsError(
          error instanceof Error
            ? error.message
            : t("benchmark.failedToLoadLocalModels"),
        );
      })
      .finally(() => {
        if (controller.signal.aborted) {
          return;
        }
        setIsLoadingLocalModels(false);
      });
    return () => controller.abort();
  }, [t]);

  const models = useMemo<ModelOption[]>(() => {
    const seen = new Set<string>();
    const result = localModels
      .filter((m) => {
        if (seen.has(m.id)) {
          return false;
        }
        seen.add(m.id);
        return true;
      })
      .map((m) => ({
        id: m.id,
        name: m.display_name ?? m.id,
        description:
          m.source === "hf_cache"
            ? t("benchmark.hfCache")
            : m.source === "custom"
              ? t("benchmark.customFolders")
              : t("benchmark.localDir"),
      }));
    return result;
  }, [localModels, t]);

  const loraModels = useMemo<LoraModelOption[]>(() => {
    const result: LoraModelOption[] = [];
    for (const run of trainingModels) {
      const tsMatch = run.name.match(/_(\d{10,})$/);
      const displayName = tsMatch ? run.name.slice(0, tsMatch.index) : run.name;
      const timeStr = tsMatch
        ? new Date(Number(tsMatch[1]) * 1000).toLocaleString(undefined, {
            dateStyle: "medium",
            timeStyle: "short",
          })
        : null;

      for (const cp of run.checkpoints) {
        const label = timeStr
          ? `${displayName} · ${cp.display_name} · ${timeStr}`
          : `${displayName} · ${cp.display_name}`;
        result.push({
          id: cp.path,
          name: label,
          description: run.base_model ?? undefined,
          baseModel: run.base_model ?? undefined,
          source: "training",
        });
      }
    }
    return result;
  }, [trainingModels]);

  const downloadUnsubRef = useRef<(() => void) | null>(null);
  const clearDownloadWatch = useCallback(() => {
    downloadUnsubRef.current?.();
    downloadUnsubRef.current = null;
  }, []);
  const setDownloadWatch = useCallback(
    (unsubscribe: (() => void) | null) => {
      clearDownloadWatch();
      downloadUnsubRef.current = unsubscribe;
    },
    [clearDownloadWatch],
  );

  const pick = useCallback(
    (value: string, meta: ModelSelectorChangeMeta) => {
      if (value !== selectedModel) {
        clearDownloadWatch();
      }
      setSelectedModel(value);
      setSelectedModelSource(meta.isLora ? "lora" : meta.source);
      setSelectedGgufVariant(meta.ggufVariant ?? null);
      setSelectedModelIsDownloaded(meta.isDownloaded ?? null);
      setDownloadingForBenchmark(false);
      if (value && value !== (inferenceParams.checkpoint ?? undefined)) {
        void selectModel({
          id: value,
          source: meta.source,
          isLora: meta.isLora,
          ggufVariant: meta.ggufVariant,
          isDownloaded: meta.isDownloaded,
          expectedBytes: meta.expectedBytes,
          isGguf: meta.isGguf,
          forceReload: true,
          throwOnError: true,
        });
      }
    },
    [
      selectedModel,
      inferenceParams.checkpoint,
      selectModel,
      clearDownloadWatch,
    ],
  );

  return {
    models,
    loraModels,
    modelsError: checkpointError ?? localModelsError,
    loadingModels: loadingCheckpoints || isLoadingLocalModels,
    selectedModel,
    selectedModelSource,
    selectedGgufVariant,
    selectedModelIsDownloaded,
    downloadingForBenchmark,
    loadingModel,
    residentModel: inferenceParams.checkpoint ?? null,
    activeGgufVariant,
    pick,
    setDownloadingForBenchmark,
    setSelectedModelIsDownloaded,
    setDownloadWatch,
    clearDownloadWatch,
  };
}
