// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect } from "react";
import { buildTrainingStartPayload } from "../api/mappers";
import {
  fetchParallelCapabilities,
  fetchParallelPlan,
} from "../api/parallel-plan-api";
import { parallelRequestFields, showsGpuSection } from "../lib/parallel-plan";
import { useTrainingConfigStore } from "../stores/training-config-store";
import { useTrainingParallelStore } from "../stores/training-parallel-store";
import type { ParallelOverride } from "../types/parallel";

const PLAN_DEBOUNCE_MS = 400;

/** The /start fields the planner reads (contract section 3); the rest of the form cannot move the plan. */
const PLAN_FIELDS = [
  "model_name",
  "training_type",
  "load_in_4bit",
  "batch_size",
  "max_seq_length",
  "lora_r",
  "target_modules",
  "gradient_checkpointing",
  "optim",
  "offload_layers",
  "packing",
  "is_dataset_image",
  "is_dataset_audio",
] as const;

export function buildParallelPlanRequest(
  override: ParallelOverride,
): Record<string, unknown> | null {
  const config = useTrainingConfigStore.getState();
  if (!config.selectedModel) return null;
  const payload = buildTrainingStartPayload(config, null) as unknown as Record<
    string,
    unknown
  >;
  const { capabilities } = useTrainingParallelStore.getState();
  const request: Record<string, unknown> = {};
  for (const field of PLAN_FIELDS) request[field] = payload[field];
  return { ...request, ...parallelRequestFields(override, capabilities) };
}

/** Reads the host's GPUs, then asks the backend how this run would use them. Only on a host where
 * /api/system already shows two or more training GPUs, so a single-GPU host makes no new request. */
export function useParallelPlan(systemGpuCount: number): void {
  const multiGpu = showsGpuSection(systemGpuCount);
  const planKey = useTrainingConfigStore((s) =>
    s.selectedModel
      ? JSON.stringify([
          s.selectedModel,
          s.trainingMethod,
          s.batchSize,
          s.contextLength,
          s.loraRank,
          s.targetModules,
          s.gradientCheckpointing,
          s.optimizerType,
          s.offloadLayers,
          s.packing,
          s.isDatasetImage,
          s.isDatasetAudio,
          s.modelType,
        ])
      : null,
  );
  const override = useTrainingParallelStore((s) => s.override);
  const capabilities = useTrainingParallelStore((s) => s.capabilities);

  useEffect(() => {
    if (!multiGpu) return;
    const controller = new AbortController();
    const store = useTrainingParallelStore.getState();
    if (store.capabilitiesStatus !== "ready")
      store.setCapabilities("loading", null);
    fetchParallelCapabilities(controller.signal)
      .then((result) => {
        if (controller.signal.aborted) return;
        const s = useTrainingParallelStore.getState();
        if (result.kind === "ok") s.setCapabilities("ready", result.value);
        else s.setCapabilities(result.kind, null);
      })
      .catch(() => {
        if (!controller.signal.aborted) {
          useTrainingParallelStore.getState().setCapabilities("error", null);
        }
      });
    return () => controller.abort();
  }, [multiGpu]);

  const ready =
    multiGpu &&
    capabilities !== null &&
    showsGpuSection(capabilities.gpus.length);
  useEffect(() => {
    const store = useTrainingParallelStore.getState();
    const request =
      ready && planKey !== null ? buildParallelPlanRequest(override) : null;
    if (!request) {
      store.clearPlan();
      return;
    }
    const controller = new AbortController();
    const timer = setTimeout(() => {
      store.setPlanLoading();
      fetchParallelPlan(request, controller.signal)
        .then((result) => {
          if (controller.signal.aborted) return;
          const s = useTrainingParallelStore.getState();
          if (result.kind === "ok") s.setPlan(result.value);
          else if (result.kind === "unsupported")
            s.setPlanUnavailable("unsupported");
          else s.setPlanUnavailable("error", result.message);
        })
        .catch((error: unknown) => {
          if (controller.signal.aborted) return;
          useTrainingParallelStore
            .getState()
            .setPlanUnavailable(
              "error",
              error instanceof Error ? error.message : String(error),
            );
        });
    }, PLAN_DEBOUNCE_MS);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [ready, planKey, override]);
}
