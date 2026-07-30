// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Placement sentinels and resolution rules shared by the load paths. Kept free
// of store / React imports so the contract is unit-testable.

// Manual-mode gpu_layers sentinel: -1 = Auto (hand layer + context sizing to
// llama.cpp's --fit). The Manual default; "all on GPU" is the slider's max.
export const GPU_LAYERS_AUTO = -1;

/** GPU placement a compare pane loads with.
 *
 * `own` is the pane's own per-model config; `shared` is the live-store snapshot
 * taken at Send (the settings the user pressed Send with, so a sequential load's
 * response echo cannot rewrite them mid-run).
 *
 * The diffusion runner honours an explicit layer split (#7574), so a pane's OWN
 * split is sent rather than forced to Auto. The shared snapshot is NOT inherited
 * by a diffusion pane: its layer count is bounded by whichever chat GGUF was
 * loaded at Send, and no diffusion UI can show or clear an inherited split (the
 * mode row and the layer slider are hidden for diffusion, and a saved diffusion
 * config is stripped of both). A leaked count would silently repartition the
 * model, and a leaked 0 masks its devices entirely.
 */
export function resolveComparePlacement(
  own: { gpuMemoryMode?: "auto" | "manual"; gpuLayers?: number },
  shared: { gpuMemoryMode: "auto" | "manual"; gpuLayers: number },
  isDiffusion: boolean,
): { gpuMemoryMode: "auto" | "manual"; gpuLayers: number } {
  return {
    gpuMemoryMode:
      own.gpuMemoryMode ?? (isDiffusion ? "auto" : shared.gpuMemoryMode),
    gpuLayers:
      own.gpuLayers ?? (isDiffusion ? GPU_LAYERS_AUTO : shared.gpuLayers),
  };
}
