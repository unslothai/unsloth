// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { formatGiB } from "../../../lib/memory/format";
import type { LoadedPlacement } from "../stores/chat-runtime-store";
import { CPU_FALLBACK_MESSAGE } from "../utils/mmproj-fallback";

// React-free so node:test can import it; shared by every model load path.

export interface OffloadCounts {
  offloaded?: number | null;
  total?: number | null;
  gpuMemoryMode?: string | null;
  gpuLayers?: number | null;
  /** The user's own `-ngl` in llama-server extras, which Auto mode respects. */
  offloadOverridden?: boolean | null;
  cpuFallbackReason?: string | null;
  gpuBackendUnavailable?: boolean | null;
}

export interface OffloadCountsSource {
  offloaded_layers?: number | null;
  offload_total_layers?: number | null;
  gpu_memory_mode?: string | null;
  gpu_layers?: number | null;
  offload_overridden?: boolean | null;
  cpu_fallback_reason?: string | null;
  gpu_backend_unavailable?: boolean | null;
}

/** Pick the split out of a load response, so the load paths cannot drift. */
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
  };
}

export interface OffloadWarning {
  /** Appended to the loaded title, e.g. "Qwen3 loaded, partly on CPU". */
  titleSuffix: string;
  description: string;
}

/** Why a loaded model is not fully on the GPU, or `null`. A known cpu_fallback_reason
 * wins over the counts: a recovered Vulkan crash also logs 0/M, and is not a size problem.
 * Manual pins only count with gpuLayers >= 0 (Manual + Auto layers is llama.cpp's fit). */
export function offloadWarning(counts: OffloadCounts): OffloadWarning | null {
  const {
    offloaded,
    total,
    gpuMemoryMode,
    gpuLayers,
    offloadOverridden,
    cpuFallbackReason,
    gpuBackendUnavailable,
  } = counts;
  if (cpuFallbackReason === "vulkan_startup_crash") {
    return {
      titleSuffix: " on CPU",
      description: CPU_FALLBACK_MESSAGE,
    };
  }
  // An unrecognised reason is still a reason: say nothing rather than guess at it.
  if (cpuFallbackReason) return null;
  if (typeof offloaded !== "number" || typeof total !== "number") return null;
  if (total <= 0 || offloaded >= total) return null;
  // Before the pin check: a GPU backend that failed to load is never what was asked for.
  if (offloaded <= 0 && gpuBackendUnavailable && gpuLayers !== 0) {
    return {
      titleSuffix: ", on CPU",
      description:
        "The GPU was found but llama.cpp could not use it, so the model runs " +
        "entirely on CPU and generation will be slow. This is a backend " +
        "problem rather than a size one, so a smaller quantization will not " +
        "help. The llama-server log in Settings > Logs has the reason.",
    };
  }
  const manualPin =
    gpuMemoryMode === "manual" &&
    typeof gpuLayers === "number" &&
    gpuLayers >= 0;
  if (manualPin || offloadOverridden) return null;
  if (offloaded <= 0) {
    return {
      titleSuffix: ", on CPU",
      description: `None of the ${total} layers fit on the GPU, so the model runs entirely on CPU and generation will be slow. A smaller quantization or a shorter context may leave room on the GPU.`,
    };
  }
  return {
    titleSuffix: ", partly on CPU",
    description: `${offloaded} of ${total} layers are on the GPU. The rest run on CPU, so generation will be slower. A smaller quantization or a shorter context may let more of it fit on the GPU.`,
  };
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

function layerCount(n: number): string {
  return `${n} ${n === 1 ? "layer" : "layers"}`;
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
  };
  return placement.total === null &&
    placement.gpuWeightsMib === null &&
    placement.hostWeightsMib === null
    ? null
    : placement;
}

/** An overflowing layer has expert weights in RAM: all its routed experts, except --fit's first partial layer
 *  per device, which keeps only part of its FFN there, so the text says expert weights, not experts. */
function describeLayers(p: LoadedPlacement): string | null {
  const { offloaded, total, overflowLayers, expertCacheMib } = p;
  if (offloaded === null || total === null || total <= 0) return null;
  // A GPU build whose backend failed to start still logs "offloaded N/N"; its buffers are all host ones.
  const noGpuBuffers = p.gpuWeightsMib === 0 && (p.hostWeightsMib ?? 0) > 0;
  if (offloaded <= 0 || noGpuBuffers) return `All ${total} layers on the CPU`;
  if (offloaded < total)
    return `${offloaded} of ${total} layers on the GPU, ${total - offloaded} on the CPU`;
  if (expertCacheMib !== null && expertCacheMib > 0) {
    return `All ${total} layers on the GPU, all experts in RAM (${weightAmount(expertCacheMib)} expert cache on the GPU)`;
  }
  if (overflowLayers !== null && overflowLayers > 0) {
    return `All ${total} layers on the GPU, expert weights of ${layerCount(overflowLayers)} in RAM`;
  }
  return `All ${total} layers on the GPU`;
}

/** Both figures are shown: token embeddings stay in RAM even when every layer is on the GPU. On a machine whose
 *  GPU and CPU share one pool, `pool` names it ("unified memory") and the weights are one figure. */
function describeWeights(
  gpuWeightsMib: number | null,
  hostWeightsMib: number | null,
  pool: string | null,
): string | null {
  const gpu = gpuWeightsMib !== null && gpuWeightsMib > 0 ? gpuWeightsMib : 0;
  const host =
    hostWeightsMib !== null && hostWeightsMib > 0 ? hostWeightsMib : 0;
  if (pool !== null && gpu + host > 0)
    return `${weightAmount(gpu + host)} in ${pool}`;
  const parts: string[] = [];
  if (gpu > 0) parts.push(`${weightAmount(gpu)} in VRAM`);
  if (host > 0) parts.push(`${weightAmount(host)} in RAM`);
  return parts.length > 0 ? parts.join(" · ") : null;
}

export function describePlacement(
  p: LoadedPlacement,
  pool: string | null = null,
): { layers: string | null; memory: string | null } {
  return {
    layers: describeLayers(p),
    memory: describeWeights(p.gpuWeightsMib, p.hostWeightsMib, pool),
  };
}
