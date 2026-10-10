// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Kept free of store and React imports so it stays unit-testable.

// Manual-mode gpu_layers sentinel: -1 = Auto, letting llama.cpp --fit size layers and context.
export const GPU_LAYERS_AUTO = -1;

export function shouldHydrateGpuPlacementControls(
  cpuFallbackReason: "vulkan_startup_crash" | null | undefined,
): boolean {
  return cpuFallbackReason !== "vulkan_startup_crash";
}

/** A diffusion pane never inherits the shared snapshot: a leaked layer count masks devices. */
export function resolveComparePlacement(
  own: { gpuMemoryMode?: "auto" | "manual"; gpuLayers?: number },
  shared: { gpuMemoryMode: "auto" | "manual"; gpuLayers: number },
  treatAsDiffusion: boolean,
): { gpuMemoryMode: "auto" | "manual"; gpuLayers: number } {
  return {
    gpuMemoryMode:
      own.gpuMemoryMode ?? (treatAsDiffusion ? "auto" : shared.gpuMemoryMode),
    gpuLayers:
      own.gpuLayers ?? (treatAsDiffusion ? GPU_LAYERS_AUTO : shared.gpuLayers),
  };
}

/** Unclassified GGUFs count: an undownloaded one reports `diffusion_unknown` but may be diffusion. */
export function shouldPinDiffusionPlacement(
  targetIsGguf: boolean,
  isDiffusion: boolean | undefined,
  diffusionUnknown: boolean,
): boolean {
  if (!targetIsGguf) return false;
  return isDiffusion === true || diffusionUnknown;
}

/** Split an older shim dropped, restored from `diffusion_requested_ngl`. Zero is a real ask. */
export function recoverDroppedDiffusionSplit(
  isDiffusion: boolean | undefined,
  mode: "auto" | "manual",
  requestedNgl: number | null | undefined,
): number | null {
  if (isDiffusion !== true || mode === "manual") return null;
  return requestedNgl ?? null;
}

/** undefined means unknown; never collapse it to false or the pane inherits another split. */
export function resolveStagedDiffusionClassification(
  knownDiffusion: boolean | undefined,
  staged:
    | { isDiffusion?: boolean; diffusionUnknown?: boolean }
    | null
    | undefined,
): boolean | undefined {
  if (knownDiffusion) return true;
  if (staged == null || staged.diffusionUnknown) return undefined;
  return staged.isDiffusion;
}
