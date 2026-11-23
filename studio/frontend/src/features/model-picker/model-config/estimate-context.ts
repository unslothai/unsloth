// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** The context to PRICE: the pin the load sends, else 0, which the backend sizes itself; never the
 *  displayed fallback or the resident's old fit. Import-free for node strip-types tests. */
export function resolveEstimateContext(contextPin: number | null): number {
  return contextPin && contextPin > 0 ? contextPin : 0;
}

/** The resident context a same-model GGUF reload keeps on Auto under Manual fixed layers, where
 *  `--fit off` would launch at the native context; null when it keeps none. `isGguf` is false and
 *  `activeLoadedContext` null for any load that is not that reload. */
export function fixedLayerContextPin({
  isGguf,
  gpuMemoryMode,
  gpuLayers,
  customContextLength,
  activeLoadedContext,
}: {
  isGguf: boolean;
  gpuMemoryMode: "auto" | "manual";
  gpuLayers: number | null | undefined;
  customContextLength: number | null | undefined;
  activeLoadedContext: number | null | undefined;
}): number | null {
  const fixedLayers = gpuMemoryMode === "manual" && gpuLayers != null && gpuLayers >= 0;
  return isGguf &&
    fixedLayers &&
    customContextLength == null &&
    activeLoadedContext != null &&
    activeLoadedContext > 0
    ? activeLoadedContext
    : null;
}

/** What the Auto context field quotes for a resident GGUF: `loaded` is the window the next load
 *  keeps (unchanged settings, or a fixed-layer reload pin); otherwise `refits` names the loaded
 *  window that Auto will fit again. */
export function autoContextQuote({
  atBaseline,
  activeLoadedContext,
  fixedLayerPin,
}: {
  atBaseline: boolean;
  activeLoadedContext: number | null;
  fixedLayerPin: number | null;
}): { loaded: number | null; refits: number | null } {
  const loaded = atBaseline ? activeLoadedContext : fixedLayerPin;
  return { loaded, refits: loaded == null ? activeLoadedContext : null };
}

export function resolveMlxServedWindow(
  loadedContext: number | null,
  fittedContext: number | null,
  nativeWindow: number | null,
): number | null {
  return fittedContext ?? loadedContext ?? nativeWindow;
}

/** Hashes the token (djb2) so the credential stays out of React keys; collisions only stale a byte count. */
export function resolveTokenIdentity(
  token: string | null | undefined,
): string {
  if (!token) return "";
  let hash = 5381;
  for (let i = 0; i < token.length; i++) {
    hash = ((hash << 5) + hash + token.charCodeAt(i)) | 0;
  }
  return (hash >>> 0).toString(36);
}

export function resolveEstimateSourceIdentity(
  modelPath: string,
  ggufVariant: string | null | undefined,
  tokenIdentity: string,
  nativePathToken: string | null | undefined,
): string {
  return JSON.stringify([
    modelPath,
    ggufVariant ?? null,
    tokenIdentity,
    nativePathToken ?? null,
  ]);
}

export function shouldRequestMemoryEstimate(opts: {
  isGguf: boolean;
  isAppleUnifiedMemory: boolean;
  classifiedIsDiffusion: boolean | undefined;
}): boolean {
  const { isGguf, isAppleUnifiedMemory, classifiedIsDiffusion } = opts;
  if (isGguf) return classifiedIsDiffusion === false;
  return isAppleUnifiedMemory && classifiedIsDiffusion !== true;
}
