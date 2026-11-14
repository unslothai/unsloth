// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** The context to PRICE, not the displayed one: an unset length loads as 0 (native/fit), so pricing
 *  the 32k display fallback understates the KV cache. Import-free for node strip-types tests. */
export function resolveEstimateContext(
  customContextLength: number | null,
  activeLoadedContext: number | null,
  skipResidentFallback = false,
): number {
  if (skipResidentFallback) {
    // Manual + Auto layers and builtin presets both resolve to 0; pricing the resident context quotes the old fit.
    return customContextLength && customContextLength > 0 ? customContextLength : 0;
  }
  return customContextLength ?? activeLoadedContext ?? 0;
}

export function resolveMlxEstimateContext(contextPin: number | null): number {
  return contextPin && contextPin > 0 ? contextPin : 0;
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
