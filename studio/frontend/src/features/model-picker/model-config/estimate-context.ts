// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The context the memory estimate should PRICE, which is not always the one the
 * Context Length control DISPLAYS.
 *
 * The control must show a number before a new GGUF's header has been read and falls
 * back to 32,768. The Load button does not: an unset length is sent as 0 and llama.cpp
 * fits, or opens at the model's native context. Pricing the displayed fallback quotes
 * an explicit 32k for a load that may open at 262k, understating the KV cache -- the
 * term that grows fastest with context. 0 is the request's own "price the native
 * context", resolved from the header exactly as the launch would.
 *
 * The fallbacks are `resolveLoadMaxSeqLength`'s, in its order and no further: an
 * explicit length is itself, reloading the resident GGUF keeps the context it is
 * resident AT, every other GGUF case is 0. The native context is deliberately NOT a
 * fallback -- --fit can land below it, so quoting it claims an outcome the load has
 * not reached.
 *
 * Import-free so `tests/` can load it under `node --experimental-strip-types`, which
 * does not resolve the `@/` alias.
 */
export function resolveEstimateContext(
  customContextLength: number | null,
  activeLoadedContext: number | null,
  skipResidentFallback = false,
): number {
  if (skipResidentFallback) {
    // Two shapes reach here, and `resolveLoadMaxSeqLength` answers 0 for both before
    // considering the resident context: Manual mode with GPU Layers on Auto, where
    // --fit owns the sizing, and a builtin-default preset on a GGUF load. In both,
    // pricing what is loaded RIGHT NOW quotes the OLD fit -- precisely when a
    // context-sensitive setting has just changed and the next fit lands elsewhere.
    return customContextLength && customContextLength > 0
      ? customContextLength
      : 0;
  }
  return customContextLength ?? activeLoadedContext ?? 0;
}

/**
 * Which MODEL an estimate belongs to, for deciding whether shown numbers still do.
 *
 * Not the same question as "did anything change" -- that is the full request key,
 * and a slider step must keep the old figures up and mark them stale rather than
 * blank the row. This is the narrower one: a change here means the numbers on
 * screen describe a DIFFERENT file, so they have to go, not go grey.
 *
 * The quantization is part of it, which is the whole reason this exists. Switching
 * Q4_K_M to F16 on the same repository leaves `modelPath` untouched while the
 * weights roughly quadruple, so keying on the path alone left one quant's footprint
 * displayed under another's name -- the exact thing the caller says must not happen.
 * The token identity and the native path token are here for the same reason: both
 * select which file the backend resolves.
 */
/**
 * Identity of a token without the token.
 *
 * The backend resolves and caches gated repositories per token, so swapping one for
 * another has to re-fetch, but the credential itself has no business sitting in a
 * React dependency key.
 *
 * djb2, because this only has to separate two strings held in this tab, never to
 * resist anyone, and it has to be synchronous (SubtleCrypto is not). It is 32 bits, so
 * a collision is constructible: two colliding credentials would leave one's estimate
 * on screen under the other, because the same value keys both the refetch and the
 * blank-the-row check. The bound is the number of tokens compared IN ONE TAB, which is
 * two -- the one that was set and the one that replaced it -- so the odds are ~2^-32
 * per swap and the consequence is a stale byte count on a row already labelled Beta,
 * not an incorrect load. Widening it is not worth a second hash; noting it is.
 */
export function resolveTokenIdentity(token: string | null | undefined): string {
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
