// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** The context the memory estimate should PRICE, which is not always the one the Context Length
 *  control DISPLAYS. The control must show a number before a new GGUF's header has been read and
 *  falls back to 32,768; the Load button does not, sending an unset length as 0 so llama.cpp
 *  fits or opens at the native context. Pricing the displayed fallback quotes an explicit 32k
 *  for a load that may open at 262k, understating the KV cache. The fallbacks are
 *  `resolveLoadMaxSeqLength`'s, in its order and no further; the native context is deliberately
 *  NOT one, since --fit can land below it. Import-free so `tests/` can load it under
 *  `node --experimental-strip-types`, which does not resolve the `@/` alias. */
export function resolveEstimateContext(
  customContextLength: number | null,
  activeLoadedContext: number | null,
  skipResidentFallback = false,
): number {
  if (skipResidentFallback) {
    // Two shapes reach here, and `resolveLoadMaxSeqLength` answers 0 for both before considering the
    // resident context: Manual mode with GPU Layers on Auto, where --fit owns the sizing, and a
    // builtin-default preset on a GGUF load. In both, pricing what is loaded RIGHT NOW quotes the
    // OLD fit, precisely when a context-sensitive setting has just changed.
    return customContextLength && customContextLength > 0 ? customContextLength : 0;
  }
  return customContextLength ?? activeLoadedContext ?? 0;
}

/** The context an MLX estimate should PRICE: the pin, else nothing. `resolveLoadMaxSeqLength`
 *  sends 0 for an unpinned MLX load and the backend answers by fitting the window to the machine.
 *  Sending the length the control happens to be DISPLAYING prices the model's native window for a
 *  load that opens narrower, and leaves the backend unable to tell a pin from a display fallback,
 *  so it could never report the fitted length this control exists to show. */
export function resolveMlxEstimateContext(contextPin: number | null): number {
  return contextPin && contextPin > 0 ? contextPin : 0;
}

/** The window the Context Length control states for a backend that sizes itself. This control
 *  describes the NEXT load, so the fit comes first: it was computed for the configuration now
 *  staged. The resident length is second and stands in wherever the estimate has no answer, but
 *  must not outrank the fit, or clearing a pin leaves the control stating the window of the
 *  pinned load it replaced. The model's own window is last, an overstatement wherever a fit lands
 *  below it. The pin is not here: it outranks all three and is applied by the caller, so a stale
 *  fitted length cannot re-enter a control the user has since set. */
export function resolveMlxServedWindow(
  loadedContext: number | null,
  fittedContext: number | null,
  nativeWindow: number | null,
): number | null {
  return fittedContext ?? loadedContext ?? nativeWindow;
}

/** Which MODEL an estimate belongs to, for deciding whether shown numbers still do. Not the same
 *  question as "did anything change" -- that is the full request key, and a slider step must
 *  keep the old figures up and mark them stale. A change here means the numbers describe a
 *  DIFFERENT file, so they have to go. The quantization is part of it: switching Q4_K_M to F16
 *  on one repository leaves `modelPath` untouched while the weights roughly quadruple. The
 *  token identity and native path token are here for the same reason. */
/** Identity of a token without the token. The backend resolves and caches gated repositories per
 *  token, so swapping one has to re-fetch, but the credential itself has no business in a React
 *  dependency key. djb2, because this only has to separate two strings held in this tab and has
 *  to be synchronous. It is 32 bits, so a collision is constructible, but the bound is the
 *  number of tokens compared IN ONE TAB, which is two, so the odds are ~2^-32 per swap and the
 *  consequence is a stale byte count on a Beta row, not an incorrect load. */
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

/**
 * Whether a prospective load should be priced at all.
 *
 * The two arms read `classifiedIsDiffusion` differently on purpose: the probe is GGUF-only, so
 * GGUF can wait for a definite `false`, while on the MLX arm one may never arrive and waiting
 * would hide the row.
 */
export function shouldRequestMemoryEstimate(opts: {
  isGguf: boolean;
  isAppleUnifiedMemory: boolean;
  classifiedIsDiffusion: boolean | undefined;
}): boolean {
  const { isGguf, isAppleUnifiedMemory, classifiedIsDiffusion } = opts;
  if (isGguf) return classifiedIsDiffusion === false;
  return isAppleUnifiedMemory && classifiedIsDiffusion !== true;
}
