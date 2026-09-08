// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * What the running llama-server reports about exact concurrency: with LLAMA_EXACT_CONCURRENCY
 * a chat gets the same tokens whether it decodes alone or beside others in one unified KV
 * cache, which is otherwise not true. Plain `.ts` and free of imports, because the test runner
 * strips types but does not transform JSX.
 */

/** `on` the server was launched with it and came up; `off` it was not asked for; `unavailable`
 *  it was asked for under `auto` and the server refused. */
export type ExactConcurrencyState = "on" | "off" | "unavailable";

/** `off` for anything unrecognised, including the `undefined` a backend older than the switch
 *  sends: the absence of an answer is not a guarantee. */
export function normalizeExactConcurrency(
  value: string | null | undefined,
): ExactConcurrencyState {
  return value === "on" || value === "unavailable" ? value : "off";
}

/** Shared by both states: the chip is only worth reading if it says what the mode buys. */
const EXACT_MEANING =
  "identical output regardless of other chats sharing this model";

/** What the header chip shows, or null when there is nothing to say. `off` is the default and
 *  the common case, so a chip for it would be noise on every load. */
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
      // Named as the server's refusal rather than a Studio failure: the setting was honoured
      // and llama-server declined it.
      title: `Exact concurrency was requested (${EXACT_MEANING}), but llama-server refused it, so this model is running without it.`,
    };
  }
  return null;
}

/** Whether the chip describes the model the user is looking at. The state is the local
 *  llama-server's, and it stays resident when a hosted model is selected beside it, so
 *  unconditionally the chip put the local server's guarantee next to a hosted model's output. */
export function exactConcurrencyChipApplies({
  isExternalModel,
  residentCheckpoint,
  modelLoading = false,
}: {
  isExternalModel: boolean;
  residentCheckpoint: string | null | undefined;
  modelLoading?: boolean;
}): boolean {
  return !isExternalModel && !modelLoading && residentCheckpoint != null;
}
