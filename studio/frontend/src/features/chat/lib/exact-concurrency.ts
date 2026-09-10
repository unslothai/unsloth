// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * What the running llama-server reports about exact concurrency: with LLAMA_EXACT_CONCURRENCY a
 * chat gets the same tokens whether it decodes alone or beside others in one unified KV cache,
 * which is otherwise not true. Plain `.ts` and free of imports, because the test runner strips
 * types but does not transform JSX.
 */

/** `on` the server was launched with it and came up; `off` it was not asked for; `unavailable` it
 *  was asked for under `auto` and this load is not running with it. */
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

/** Appended when the server said this thread took a recompute: the mode is running, but one
 *  answer here did not get the guarantee, and a chip that stayed silent would overstate it. */
const RECOMPUTED_NOTE =
  "This answer was re-prefilled after a park the server could not hold, so it is not guaranteed byte-identical.";

/** What the header chip shows, or null when there is nothing to say. `off` is the default and
 *  the common case, so a chip for it would be noise on every load. `recomputed` is this thread's,
 *  not the load's: the server reports it per answer. */
export function exactConcurrencyChip(
  state: ExactConcurrencyState,
  options: { recomputed?: boolean } = {},
): { label: string; title: string } | null {
  const note = options.recomputed ? ` ${RECOMPUTED_NOTE}` : "";
  if (state === "on") {
    return {
      label: options.recomputed ? "Exact, one answer re-prefilled" : "Exact",
      title: `Exact concurrency is on: ${EXACT_MEANING}.${note}`,
    };
  }
  if (state === "unavailable") {
    return {
      label: "Exact unavailable",
      // Not "refused": Studio withholds the mode itself on some launches, and the load
      // warnings carry the reason.
      title: `Exact concurrency was requested (${EXACT_MEANING}), but this load is not running with it, so a chat's output can depend on the other chats sharing this model.${note}`,
    };
  }
  return null;
}

/** Whether the chip describes the model the user is looking at: the state is the local
 *  llama-server's, and it stays resident when a hosted model is selected beside it. */
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
