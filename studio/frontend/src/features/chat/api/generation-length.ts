// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Whether Max Tokens, rather than the context window, is what stopped this generation. Two
 *  different walls produce the same `finish_reason: "length"`, and only one has a setting the user
 *  can raise. Comparing the cap to the whole window answers the wrong question: with a 4096-token
 *  window, a 3000-token prompt and Max Tokens 2048, generation stops after roughly 1096 tokens,
 *  well short of the cap, and "increase Max Tokens" cannot create any room. With Max Tokens on
 *  "Max" the backend sends the whole context length, so a cap equal to it is indistinguishable
 *  from unset. `promptTokens` is the server's own count from the final usage chunk; absent, the
 *  cap alone decides. */
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
  // Strictly below, not at. At equality the cap and the physical context wall are hit in the same
  // token, so raising Max Tokens creates no room. The context length is the lever there.
  return promptTokens + cap < window;
}
