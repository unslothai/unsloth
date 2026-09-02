// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Whether Max Tokens, rather than the context window, is what stopped this generation.
 *
 * Two different walls produce the same `finish_reason: "length"`, and only one of them has
 * a setting the user can raise. Comparing the cap to the whole window answers the first
 * question and not the second: with a 4096-token window, a 3000-token prompt and Max
 * Tokens 2048, generation stops after roughly 1096 tokens -- well short of the cap -- and
 * "increase Max Tokens" is then advice that cannot create any room at all.
 *
 * With Max Tokens on "Max" the backend sends the whole context length, so a cap equal to
 * it is indistinguishable from unset here, and both mean the same thing to the user.
 *
 * `promptTokens` is the server's own count from the final usage chunk. Absent, the cap
 * alone decides, which is what this did before the prompt was available.
 */
export function maxTokensIsTheLimit({
  cap,
  contextLength,
  promptTokens,
}: {
  cap: number | null;
  contextLength: number | null;
  promptTokens: number | null;
}): boolean {
  const window = contextLength ?? Number.POSITIVE_INFINITY;
  if (cap === null || cap >= window) {
    return false;
  }
  if (promptTokens === null) {
    return true;
  }
  // Strictly below, not at. At equality the cap and the physical context wall are hit
  // in the same token, so raising Max Tokens creates no room and the advice to do so is
  // the one thing that cannot help. The context length is the lever there.
  return promptTokens + cap < window;
}
