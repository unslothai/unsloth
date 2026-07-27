// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * One diffusion (DiffusionGemma) classification for the whole app.
 *
 * The model picker gates its controls on it and the chat API layer gates what it
 * sends on it, so it lives here, dependency-free, rather than inside either
 * feature: importing it must never drag a feature barrel into the request path.
 */

// Mirror the backend's _classify_diffusion_gguf (routes/inference.py): with no header,
// strip non-alphanumerics and look for the DiffusionGemma name so staged Load agrees.
export function looksLikeDiffusionGemma(
  ...parts: (string | null | undefined)[]
): boolean {
  return parts.some((part) =>
    (part ?? "")
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "")
      .includes("diffusiongemma"),
  );
}

// Tri-state, mirroring the backend's _preflight_is_diffusion (core/inference/llama_cpp.py):
// null means no GGUF header has been read yet, so nothing is classified.
export type DiffusionClassification = boolean | null;

// Resolve one diffusion answer from the backend's header classification plus the name.
// The header wins whenever it exists: it is what /validate and /load themselves decide
// on, so an ordinary GGUF that merely carries DiffusionGemma in its id keeps the controls
// those endpoints accept for it. The name check is only the stand-in for a staged pick,
// whose header the backend has not read yet either.
export function resolveIsDiffusion(
  loaded: DiffusionClassification,
  ...parts: (string | null | undefined)[]
): boolean {
  return loaded ?? looksLikeDiffusionGemma(...parts);
}

/**
 * Drop a host-memory mode the diffusion runner has no equivalent for.
 *
 * ``_reject_diffusion_memory_mode`` 400s an explicit ``gguf_memory_mode`` on both
 * /validate and /load, and the control that would clear it is hidden for such a
 * model, so a remembered mode would fail every load with no visible way out. This
 * runs on what is SENT only: a mode that stays in storage is valid again the moment
 * the same id loads with a header saying it is not diffusion, which is exactly the
 * case ``loaded`` carries and the name check cannot tell apart.
 */
export function diffusionSafeMemoryMode<Mode>(
  memoryMode: Mode | null | undefined,
  loaded: DiffusionClassification,
  ...parts: (string | null | undefined)[]
): Mode | null {
  if (memoryMode == null) {
    return null;
  }
  return resolveIsDiffusion(loaded, ...parts) ? null : memoryMode;
}
