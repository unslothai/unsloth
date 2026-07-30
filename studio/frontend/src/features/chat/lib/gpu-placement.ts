// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Placement sentinels and resolution rules shared by the load paths. Kept free
// of store / React imports so the contract is unit-testable.

// Manual-mode gpu_layers sentinel: -1 = Auto (hand layer + context sizing to
// llama.cpp's --fit). The Manual default; "all on GPU" is the slider's max.
export const GPU_LAYERS_AUTO = -1;

/** GPU placement a compare pane loads with.
 *
 * `own` is the pane's per-model config; `shared` is the live-store snapshot taken
 * at Send, so a sequential load's response echo cannot rewrite it mid-run.
 *
 * A pane's OWN split is sent rather than forced to Auto, since the diffusion
 * runner honours it (#7574). The shared snapshot is NOT inherited by a diffusion
 * pane: its layer count is bounded by another GGUF, no diffusion UI can show or
 * clear it, and a leaked 0 masks the devices entirely.
 */
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

/** Whether a pane must get the diffusion-safe placement above.
 *
 * An UNCLASSIFIED GGUF counts: the preflight only reads a header already on disk,
 * so an undownloaded GGUF with no family in its name comes back
 * `diffusion_unknown`, and `/load` may then read a diffusion header and apply the
 * inherited count anyway.
 *
 * Non-GGUF panes are excluded: definitively not diffusion, and they send no
 * placement at all.
 */
export function shouldPinDiffusionPlacement(
  targetIsGguf: boolean,
  isDiffusion: boolean | undefined,
  diffusionUnknown: boolean,
): boolean {
  if (!targetIsGguf) return false;
  return isDiffusion === true || diffusionUnknown;
}
