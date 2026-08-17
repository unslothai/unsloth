// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** What a load's layer split is worth telling the user, from llama.cpp's own count.
 *
 * Kept free of React so it can be tested: the runtime hook pulls in the router and
 * the whole chat store and cannot be imported under node:test. Shared, because four
 * separate flows load a model (the model picker, a chat sent with nothing loaded,
 * compare mode, and a recipe run) and each used to announce success on its own.
 *
 * Studio only chooses this split itself in Manual mode. On Auto it decides the model
 * does not provably fit and hands placement to llama.cpp's `--fit on`, which quietly
 * puts some or all layers on the CPU. Decode then runs at a fraction of the speed,
 * and the load reported success and nothing else, so the only way to find out was to
 * notice the model was slow.
 */

export interface OffloadCounts {
  offloaded?: number | null;
  total?: number | null;
  /** "manual" means the user picked the layer count in the load panel. */
  gpuMemoryMode?: string | null;
  /** The requested count. -1 is Auto, which Manual mode also allows. */
  gpuLayers?: number | null;
  /** The user's own `-ngl` in llama-server extras, which Auto mode respects. */
  offloadOverridden?: boolean | null;
  /** Set when the backend already knows why the model is on the CPU. */
  cpuFallbackReason?: string | null;
  /** Studio found a GPU for this load but llama.cpp reported none. */
  gpuBackendUnavailable?: boolean | null;
}

/** The snake_case load response, narrowed to the fields this file reads. */
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

/** Why a loaded model is not fully on the GPU, or `null` when it is or nobody cares.
 *
 * A known reason wins over the counts. A recovered Vulkan startup crash leaves the
 * model on the CPU with a `0/M` line behind it, and reading that as "nothing fit"
 * would blame the model's size for a backend crash and recommend a quantization
 * that changes nothing.
 *
 * `null` for a placement the user chose. Manual mode with an explicit layer count is
 * the obvious one, but Manual with GPU Layers left on Auto hands placement back to
 * llama.cpp exactly like Auto mode does, so the mode alone is not the question: the
 * requested count is. The other is an `-ngl` passed through llama-server extras,
 * which Auto mode deliberately respects rather than strips. In both cases "a smaller
 * quantization would fit" is advice against their own choice.
 *
 * `null` too when the counts are missing, which the backend also uses to say the
 * split is not reportable: an MLX or transformers load, a llama.cpp build whose log
 * did not say, or a host with no GPU at all, where llama.cpp still logs `0/N`.
 * Every layer on the GPU is the normal case and needs no notice either.
 *
 * Zero offloaded layers is otherwise a warning, not an exclusion. It is the worst
 * version of this problem and, absent a known reason, it had no reporting at all:
 * an ordinary `--fit on` load that got nothing onto the GPU announced plain success.
 */
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
      description:
        "The auto-selected Vulkan backend crashed during startup, so GPU " +
        "acceleration is disabled for this model session.",
    };
  }
  // An unrecognised reason is still a reason: say nothing rather than guess at it.
  if (cpuFallbackReason) return null;
  const manualPin =
    gpuMemoryMode === "manual" &&
    typeof gpuLayers === "number" &&
    gpuLayers >= 0;
  if (manualPin || offloadOverridden) return null;
  if (typeof offloaded !== "number" || typeof total !== "number") return null;
  if (total <= 0 || offloaded >= total) return null;
  if (offloaded <= 0) {
    // Nothing on the GPU has two causes that log the same 0/M line, and telling a
    // user with a broken CUDA install to pick a smaller quantization sends them
    // to re-download a model that was never the problem.
    if (gpuBackendUnavailable) {
      return {
        titleSuffix: ", on CPU",
        description:
          "The GPU was found but llama.cpp could not use it, so the model runs " +
          "entirely on CPU and generation will be slow. This is a backend " +
          "problem rather than a size one, so a smaller quantization will not " +
          "help. The llama-server log in Settings > Logs has the reason.",
      };
    }
    return {
      titleSuffix: ", on CPU",
      description:
        `None of the ${total} layers fit on the GPU, so the model runs entirely on ` +
        "CPU and generation will be slow. A smaller quantization would leave room " +
        "on the GPU.",
    };
  }
  return {
    titleSuffix: ", partly on CPU",
    description:
      `${offloaded} of ${total} layers are on the GPU. The rest run on CPU, so ` +
      "generation will be slower. A smaller quantization would fit entirely on " +
      "the GPU.",
  };
}
