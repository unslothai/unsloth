// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** The context to PRICE: an unset length loads as 0 (native/fit), so neither the 32k display
 *  fallback nor the resident's old fit is the figure. Import-free for node strip-types tests. */
export function resolveEstimateContext(customContextLength: number | null): number {
  return customContextLength && customContextLength > 0 ? customContextLength : 0;
}

/** The resident context a same-model GGUF reload keeps on Auto under Manual fixed layers, where
 *  `--fit off` would launch at the native context; null when it keeps none. */
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
