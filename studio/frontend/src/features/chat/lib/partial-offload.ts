// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { formatGiB } from "../../../lib/memory/format";
import type { Translate } from "./generation-progress";
import type { LoadedPlacement } from "../stores/chat-runtime-store";
import { cpuFallbackMessage } from "../utils/mmproj-fallback";

// React-free so node:test can import it.

export interface OffloadCounts {
  offloaded?: number | null;
  total?: number | null;
  gpuMemoryMode?: string | null;
  gpuLayers?: number | null;
  offloadOverridden?: boolean | null;
  cpuFallbackReason?: string | null;
  gpuBackendUnavailable?: boolean | null;
  overflowLayers?: number | null;
  expertCacheMib?: number | null;
  gpuWeightsMib?: number | null;
  hostWeightsMib?: number | null;
  metalUnsupportedWeightTypes?: string[] | null;
}

export interface OffloadCountsSource {
  offloaded_layers?: number | null;
  offload_total_layers?: number | null;
  gpu_memory_mode?: string | null;
  gpu_layers?: number | null;
  offload_overridden?: boolean | null;
  cpu_fallback_reason?: string | null;
  gpu_backend_unavailable?: boolean | null;
  overflow_layers?: number | null;
  expert_cache_mib?: number | null;
  gpu_weights_mib?: number | null;
  host_weights_mib?: number | null;
  metal_unsupported_weight_types?: string[] | null;
}

export function offloadCountsFrom(
  response: OffloadCountsSource,
): OffloadCounts {
  return {
    offloaded: response.offloaded_layers,
    total: response.offload_total_layers,
    gpuMemoryMode: response.gpu_memory_mode,
    gpuLayers: response.gpu_layers,
    offloadOverridden: response.offload_overridden,
    cpuFallbackReason: response.cpu_fallback_reason,
    gpuBackendUnavailable: response.gpu_backend_unavailable,
    overflowLayers: response.overflow_layers,
    expertCacheMib: response.expert_cache_mib,
    gpuWeightsMib: response.gpu_weights_mib,
    hostWeightsMib: response.host_weights_mib,
    metalUnsupportedWeightTypes: response.metal_unsupported_weight_types,
  };
}

/** "NVFP4", "NVFP4 and TQ1_0": llama.cpp's type names, upper-cased; null for none. */
function weightTypeList(types: readonly unknown[] | null | undefined): string | null {
  const names = (types ?? [])
    .filter((t): t is string => typeof t === "string" && t.length > 0)
    .map((t) => t.toUpperCase());
  if (names.length === 0) return null;
  return names.length === 1
    ? names[0]
    : `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
}

export interface OffloadWarning {
  titleSuffix: string;
  description: string;
}

function count(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) && value >= 0
    ? value
    : null;
}

/** GiB like the rest of the panel, or MiB under 0.1 GiB, which would otherwise print as "0.0 GiB". */
function weightAmount(mib: number): string {
  return mib < 102.4
    ? `${Math.max(1, Math.round(mib))} MiB`
    : formatGiB(mib / 1024);
}

/** Manual pins only count with gpuLayers >= 0 (Manual + Auto layers is llama.cpp's fit). */
function splitChosenByUser(counts: OffloadCounts): boolean {
  return (
    (counts.gpuMemoryMode === "manual" &&
      typeof counts.gpuLayers === "number" &&
      counts.gpuLayers >= 0) ||
    counts.offloadOverridden === true
  );
}

type ShortfallCause = "fallback" | "backend" | "metal" | "noneFit" | "partial";

/** Why a load is not fully on the GPU although nothing asked for that, or `null`. A cpu_fallback_reason
 * wins over the counts: a Vulkan CPU fallback also logs 0/M, and is not a size problem. */
function shortfallCause(counts: OffloadCounts): ShortfallCause | null {
  const { offloaded, total, gpuLayers, gpuBackendUnavailable } = counts;
  if (counts.cpuFallbackReason) return "fallback";
  if (typeof offloaded !== "number" || typeof total !== "number") return null;
  if (total <= 0) return null;
  // A GPU build whose backend failed to start still logs "offloaded N/N"; its buffers are all host ones.
  const noGpuBuffers =
    count(counts.gpuWeightsMib) === 0 && (count(counts.hostWeightsMib) ?? 0) > 0;
  const onGpu = gpuBackendUnavailable && noGpuBuffers ? 0 : offloaded;
  // No layer split, pin or context changes where llama.cpp runs these weights.
  if (onGpu > 0 && weightTypeList(counts.metalUnsupportedWeightTypes)) return "metal";
  if (onGpu >= total) return null;
  // Before the pin check: a failed GPU backend is never what was asked for.
  if (onGpu <= 0 && gpuBackendUnavailable && gpuLayers !== 0) return "backend";
  if (splitChosenByUser(counts)) return null;
  return onGpu <= 0 ? "noneFit" : "partial";
}

/** Weights only, no KV cache. Names the device, not VRAM/RAM: those are one pool on some machines. */
function weightsSentence(counts: OffloadCounts): string {
  const gpu = count(counts.gpuWeightsMib) ?? 0;
  const host = count(counts.hostWeightsMib) ?? 0;
  const parts: string[] = [];
  if (gpu > 0) parts.push(`${weightAmount(gpu)} on the GPU`);
  if (host > 0) parts.push(`${weightAmount(host)} on the CPU`);
  return parts.length > 0 ? ` Model weights: ${parts.join(", ")}.` : "";
}

export function offloadWarning(counts: OffloadCounts): OffloadWarning | null {
  const { offloaded, total } = counts;
  switch (shortfallCause(counts)) {
    case "fallback": {
      // A reason this build does not know is left to the caller's own fallback wording.
      const description = cpuFallbackMessage(counts.cpuFallbackReason);
      return description ? { titleSuffix: " on CPU", description } : null;
    }
    case "backend":
      return {
        titleSuffix: ", on CPU",
        description:
          "The GPU was found but llama.cpp could not use it, so the model runs " +
          "entirely on CPU and generation will be slow. This is a backend " +
          "problem rather than a size one, so a smaller quantization will not " +
          "help. The llama-server log in Settings > Logs has the reason.",
      };
    case "metal": {
      const types = weightTypeList(counts.metalUnsupportedWeightTypes);
      const nvfp4Only = counts.metalUnsupportedWeightTypes?.every((t) => t.toLowerCase() === "nvfp4");
      return {
        titleSuffix: ", partly on CPU",
        description: `Metal cannot run this model's ${types} weights in this llama.cpp build, so they run on the CPU although every layer counts as offloaded, and generation will be much slower. A shorter context will not help. ${nvfp4Only ? "Update llama.cpp: newer builds run NVFP4 on the GPU." : `A quantization without ${types} tensors runs fully on the GPU.`}${weightsSentence(counts)}`,
      };
    }
    case "noneFit":
      return {
        titleSuffix: ", on CPU",
        description: `None of the ${total} layers fit on the GPU, so the model runs entirely on CPU and generation will be slow. A smaller quantization or a shorter context may leave room on the GPU.${weightsSentence(counts)}`,
      };
    case "partial":
      return {
        titleSuffix: ", partly on CPU",
        description: `${offloaded} of ${total} layers are on the GPU. The rest run on CPU, so generation will be slower. A smaller quantization or a shorter context may let more of it fit on the GPU.${weightsSentence(counts)}`,
      };
    default:
      return null;
  }
}

/** Not a warning: --fit keeping MoE expert weights on the CPU is how a large MoE fits, far cheaper than layers. */
export function expertSpillNote(counts: OffloadCounts): string | null {
  const { offloaded, total } = counts;
  if (typeof offloaded !== "number" || typeof total !== "number") return null;
  if (total <= 0 || offloaded < total) return null;
  if (shortfallCause(counts) !== null || splitChosenByUser(counts)) return null;
  const cache = count(counts.expertCacheMib) ?? 0;
  const overflow = count(counts.overflowLayers) ?? 0;
  if (cache > 0) {
    return `All ${total} layers are on the GPU. Every expert is on the CPU, with a ${weightAmount(cache)} cache of the most used ones on the GPU.${weightsSentence(counts)}`;
  }
  if (overflow > 0) {
    return `All ${total} layers are on the GPU. The expert weights of ${overflow} ${overflow === 1 ? "layer" : "layers"} are on the CPU.${weightsSentence(counts)}`;
  }
  return null;
}

/** --fit reports its own expert spill; a Manual load says it through n_cpu_moe instead. */
export function loadedPlacementFrom(response: {
  is_gguf?: boolean | null;
  offloaded_layers?: number | null;
  offload_total_layers?: number | null;
  overflow_layers?: number | null;
  gpu_weights_mib?: number | null;
  host_weights_mib?: number | null;
  expert_cache_mib?: number | null;
  gpu_memory_mode?: string | null;
  n_cpu_moe?: number | null;
  metal_unsupported_weight_types?: string[] | null;
}): LoadedPlacement | null {
  if (response.is_gguf === false) return null;
  const manualExperts =
    response.gpu_memory_mode === "manual" ? count(response.n_cpu_moe) : null;
  const placement: LoadedPlacement = {
    offloaded: count(response.offloaded_layers),
    total: count(response.offload_total_layers),
    overflowLayers:
      count(response.overflow_layers) ?? (manualExperts ? manualExperts : null),
    gpuWeightsMib: count(response.gpu_weights_mib),
    hostWeightsMib: count(response.host_weights_mib),
    expertCacheMib: count(response.expert_cache_mib) || null,
    cpuWeightTypes: response.metal_unsupported_weight_types?.length
      ? response.metal_unsupported_weight_types
      : null,
  };
  return placement.total === null &&
    placement.gpuWeightsMib === null &&
    placement.hostWeightsMib === null
    ? null
    : placement;
}

/** An overflowing layer has expert weights in RAM: all its routed experts, except --fit's first partial layer
 *  per device, which keeps only part of its FFN there, so the text says expert weights, not experts. */
function describeLayers(p: LoadedPlacement, t: Translate): string | null {
  const layers = describeLayerSplit(p, t);
  const types = weightTypeList(p.cpuWeightTypes);
  return layers && types && (p.offloaded ?? 0) > 0
    ? `${layers} · ${t("generationStats.placement.metalCpuWeights", { types })}`
    : layers;
}

function describeLayerSplit(p: LoadedPlacement, t: Translate): string | null {
  const { offloaded, total, overflowLayers, expertCacheMib } = p;
  if (offloaded === null || total === null || total <= 0) return null;
  // A GPU build whose backend failed to start still logs "offloaded N/N"; its buffers are all host ones.
  const noGpuBuffers = p.gpuWeightsMib === 0 && (p.hostWeightsMib ?? 0) > 0;
  if (offloaded <= 0 || noGpuBuffers) return t("generationStats.placement.allOnCpu", { total });
  if (offloaded < total)
    return t("generationStats.placement.split", { gpu: offloaded, total, cpu: total - offloaded });
  if (expertCacheMib !== null && expertCacheMib > 0) {
    return t("generationStats.placement.expertCache", { total, amount: weightAmount(expertCacheMib) });
  }
  if (overflowLayers !== null && overflowLayers > 0) {
    return t(
      overflowLayers === 1
        ? "generationStats.placement.expertSpillOne"
        : "generationStats.placement.expertSpill",
      { total, count: overflowLayers },
    );
  }
  return t("generationStats.placement.allOnGpu", { total });
}

export type PlacementPool = "unified" | "shared";

/** Both figures are shown: token embeddings stay in RAM even when every layer is on the GPU. On a machine whose
 *  GPU and CPU share one pool, `pool` names it and the weights are one figure. */
function describeWeights(
  gpuWeightsMib: number | null,
  hostWeightsMib: number | null,
  pool: PlacementPool | null,
  t: Translate,
): string | null {
  const gpu = gpuWeightsMib !== null && gpuWeightsMib > 0 ? gpuWeightsMib : 0;
  const host =
    hostWeightsMib !== null && hostWeightsMib > 0 ? hostWeightsMib : 0;
  if (pool !== null && gpu + host > 0)
    return t(
      pool === "unified"
        ? "generationStats.placement.inUnifiedMemory"
        : "generationStats.placement.inSharedMemory",
      { amount: weightAmount(gpu + host) },
    );
  const parts: string[] = [];
  if (gpu > 0) parts.push(t("generationStats.placement.inVram", { amount: weightAmount(gpu) }));
  if (host > 0) parts.push(t("generationStats.placement.inRam", { amount: weightAmount(host) }));
  return parts.length > 0 ? parts.join(" · ") : null;
}

export function describePlacement(
  p: LoadedPlacement,
  t: Translate,
  pool: PlacementPool | null = null,
): { layers: string | null; memory: string | null } {
  return {
    layers: describeLayers(p, t),
    memory: describeWeights(p.gpuWeightsMib, p.hostWeightsMib, pool, t),
  };
}
