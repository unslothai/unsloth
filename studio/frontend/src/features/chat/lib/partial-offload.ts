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
  /** The user's own `-ngl` in llama-server extras, which Auto mode respects. */
  offloadOverridden?: boolean | null;
}

/** The snake_case load response, narrowed to the fields this file reads. */
export interface OffloadCountsSource {
  offloaded_layers?: number | null;
  offload_total_layers?: number | null;
  gpu_memory_mode?: string | null;
  offload_overridden?: boolean | null;
}

/** Pick the split out of a load response, so the four load paths cannot drift. */
export function offloadCountsFrom(
  response: OffloadCountsSource,
): OffloadCounts {
  return {
    offloaded: response.offloaded_layers,
    total: response.offload_total_layers,
    gpuMemoryMode: response.gpu_memory_mode,
    offloadOverridden: response.offload_overridden,
  };
}

export interface OffloadWarning {
  /** Appended to the loaded title, e.g. "Qwen3 loaded, partly on CPU". */
  titleSuffix: string;
  description: string;
}

/** The warning a load's split deserves, or `null` when it deserves none.
 *
 * `null` for a placement the user chose. Manual mode is the obvious one; the other
 * is an `-ngl` (or `--fit off`) they passed through llama-server extras, which Auto
 * mode deliberately respects rather than strips, so the reported mode alone cannot
 * tell a spill from a decision. In both cases "a smaller quantization would fit" is
 * advice against their own choice.
 *
 * `null` too when the counts are missing (an MLX or transformers load, or a
 * llama.cpp build whose log did not say) and when every layer made it onto the GPU,
 * which is the normal case and needs no notice.
 *
 * Zero offloaded layers is a warning, not an exclusion. It is the worst version of
 * this problem and it has no other reporting: `cpu_fallback_reason` is only ever set
 * for the Vulkan startup-crash recovery, so an ordinary `--fit on` load that got
 * nothing onto the GPU announced plain success.
 */
export function offloadWarning(counts: OffloadCounts): OffloadWarning | null {
  const { offloaded, total, gpuMemoryMode, offloadOverridden } = counts;
  if (gpuMemoryMode === "manual" || offloadOverridden) return null;
  if (typeof offloaded !== "number" || typeof total !== "number") return null;
  if (total <= 0 || offloaded >= total) return null;
  if (offloaded <= 0) {
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
