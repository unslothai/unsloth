// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The context the memory estimate should PRICE, which is not always the context
 * the Context Length control DISPLAYS.
 *
 * The control has to show a number even before a new GGUF's header has been read,
 * and falls back to 32,768 for that. The Load button does not: an unset length is
 * sent as 0 and llama.cpp fits, or opens at the model's native context. Pricing the
 * displayed fallback therefore quotes an explicit 32k for a load that may well open
 * at 262k, and understates by the KV cache, the term that grows fastest with
 * context.
 *
 * 0 is the estimate request's own "price the native context", so the backend
 * resolves it from the header exactly as the launch would.
 *
 * Kept in its own module with no imports so `tests/` can load it under
 * `node --experimental-strip-types`, which does not resolve the `@/` alias.
 */
export function resolveEstimateContext(
  customContextLength: number | null,
  activeLoadedContext: number | null,
  nativeContextLength: number | null,
): number {
  return customContextLength ?? activeLoadedContext ?? nativeContextLength ?? 0;
}
