// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface CtxPinSeed {
  customContextLength?: number | null;
  loadedCustomContextLength?: number | null;
}

const CLEAR: CtxPinSeed = {
  customContextLength: null,
  loadedCustomContextLength: null,
};

/** A positive echo is ambiguous (auto reloads send the resolved n_ctx), so never invent a pin.
 *  Exception: Manual with Auto layers sends 0 for Auto, so a positive echo is a pin. */
export function resolveCtxPinSeed(options: {
  incoming: number | null | undefined;
  /** Non-GGUF statuses report incoming too; this flag keeps them out of the pin. */
  isGguf: boolean;
  /** Unpinned MLX sends 0, so a positive echo is a pin. */
  isMlx?: boolean;
  seedLoadParams: boolean;
  modelChanged: boolean;
  remembered: number | null;
  gpuMemoryMode?: "auto" | "manual" | null;
  /** Not normalised: negative means Auto layers. */
  gpuLayers?: number | null;
  loadedPin?: number | null;
}): CtxPinSeed {
  const {
    incoming,
    isGguf,
    isMlx,
    seedLoadParams,
    modelChanged,
    remembered,
    gpuMemoryMode,
    gpuLayers,
    loadedPin,
  } = options;
  // During a load, status still answers for the outgoing model, so seed nothing.
  if (!seedLoadParams) return {};
  if (!isGguf) return CLEAR;
  if (incoming === undefined) {
    return modelChanged ? CLEAR : {};
  }
  // 0 is the wire value for Auto, so it is the one unambiguous echo.
  if (incoming === null || !(incoming > 0)) return CLEAR;
  // Unambiguous on MLX: adopt it, or the next Apply would send 0 and drop another client's pin.
  if (isMlx) {
    return { customContextLength: incoming, loadedCustomContextLength: incoming };
  }
  // Manual memory + Auto layers always sends 0 for Auto, so a positive echo proves a pin.
  if (gpuMemoryMode === "manual" && gpuLayers != null && gpuLayers < 0) {
    return { customContextLength: incoming, loadedCustomContextLength: incoming };
  }
  if (!modelChanged) {
    // Same model: keep the record unless the echo contradicts it (another client reloaded).
    return loadedPin != null && loadedPin !== incoming ? CLEAR : {};
  }
  // Model changed: re-pin only if the saved config matches the running server.
  return remembered != null && remembered > 0 && remembered === incoming
    ? {
        customContextLength: remembered,
        loadedCustomContextLength: remembered,
      }
    : CLEAR;
}
