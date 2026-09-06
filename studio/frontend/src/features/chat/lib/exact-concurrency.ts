// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * What the running llama-server reports about exact concurrency.
 *
 * A server launched with LLAMA_EXACT_CONCURRENCY gives a chat the same tokens whether it
 * decodes alone or beside other chats in the one unified KV cache, which is otherwise not
 * true: a neighbour joining or leaving changes the batch, and the batch changes the
 * arithmetic. That is invisible without being told, so the header says it.
 *
 * Plain `.ts` and free of imports on purpose: the test runner is
 * `node --experimental-strip-types`, which strips types but does not transform JSX, so
 * nothing reachable only from a `.tsx` can be unit-tested.
 */

/** `on` the server was launched with it and came up; `off` it was not asked for;
 *  `unavailable` it was asked for under `auto` and the server refused, so the load runs
 *  without the guarantee. */
export type ExactConcurrencyState = "on" | "off" | "unavailable";

/**
 * `off` for anything unrecognised, including the `undefined` a backend older than the
 * switch sends. Claiming the guarantee needs the server to have said so; the absence of
 * an answer is not one.
 */
export function normalizeExactConcurrency(
  value: string | null | undefined,
): ExactConcurrencyState {
  return value === "on" || value === "unavailable" ? value : "off";
}

/** Shared by both states: the chip is only worth reading if it says what the mode buys. */
const EXACT_MEANING =
  "identical output regardless of other chats sharing this model";

/**
 * What the header chip shows, or null when there is nothing to say. `off` is the default
 * and the common case, so a chip for it would be noise on every load.
 */
export function exactConcurrencyChip(
  state: ExactConcurrencyState,
): { label: string; title: string } | null {
  if (state === "on") {
    return {
      label: "Exact",
      title: `Exact concurrency is on: ${EXACT_MEANING}.`,
    };
  }
  if (state === "unavailable") {
    return {
      label: "Exact unavailable",
      // Named as the server's refusal rather than a Studio failure: the setting was
      // honoured, the request was made, and llama-server declined it.
      title: `Exact concurrency was requested (${EXACT_MEANING}), but llama-server refused it, so this model is running without it.`,
    };
  }
  return null;
}
