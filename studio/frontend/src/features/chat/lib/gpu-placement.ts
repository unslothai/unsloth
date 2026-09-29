// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Placement sentinels and resolution rules shared by the load paths. Kept free of store and
// React imports so the contract is unit-testable.

// Manual-mode gpu_layers sentinel: -1 = Auto, handing layer and context sizing to llama.cpp's
// --fit. The Manual default; "all on GPU" is the slider's max.
export const GPU_LAYERS_AUTO = -1;

export function shouldHydrateGpuPlacementControls(
  cpuFallbackReason: "vulkan_startup_crash" | null | undefined,
): boolean {
  return cpuFallbackReason !== "vulkan_startup_crash";
}

/** GPU placement a compare pane loads with. `own` is the pane's per-model config; `shared` is the
 *  live-store snapshot taken at Send, so a sequential load's echo cannot rewrite it mid-run. A
 *  pane's OWN split is sent rather than forced to Auto, since the diffusion runner honours it
 *  (#7574). The shared snapshot is NOT inherited by a diffusion pane: its layer count is
 *  bounded by another GGUF, no diffusion UI can clear it, and a leaked 0 masks the devices. */
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

/** Whether a pane must get the diffusion-safe placement above. An UNCLASSIFIED GGUF counts: the
 *  preflight only reads a header already on disk, so an undownloaded GGUF with no family in its
 *  name comes back `diffusion_unknown`, and `/load` may then read a diffusion header and apply
 *  the inherited count anyway. Non-GGUF panes send no placement and are excluded. */
export function shouldPinDiffusionPlacement(
  targetIsGguf: boolean,
  isDiffusion: boolean | undefined,
  diffusionUnknown: boolean,
): boolean {
  if (!targetIsGguf) return false;
  return isDiffusion === true || diffusionUnknown;
}

/** The split an older shim DROPPED, recovered from a load/status response. A shim without
 *  `--ngl` cannot apply a manual split, so the runner reports Auto while the backend keeps the
 *  ask in `diffusion_requested_ngl`. In-memory state carries that across a reload but not a
 *  refresh, so the ask has to be restored or the next Apply sends `manual/-1`, which the
 *  post-upgrade retry can never turn back into the count. Null when there is nothing to
 *  recover; zero is a real ask (CPU-only) and is preserved. */
export function recoverDroppedDiffusionSplit(
  isDiffusion: boolean | undefined,
  mode: "auto" | "manual",
  requestedNgl: number | null | undefined,
): number | null {
  if (isDiffusion !== true || mode === "manual") return null;
  return requestedNgl ?? null;
}

/** Tri-state diffusion classification for a staged (pre-load) GGUF selection. `undefined` means
 *  "not known" and must NOT be collapsed to false by a caller passing the answer on: a definite
 *  false tells the compare flow this is an ordinary GGUF, skipping the re-probe and letting an
 *  unconfigured pane inherit another model's layer split (#7574). */
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
