// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Exact concurrency as the running llama-server reports it. Plain `.ts` with no imports: the test
 *  runner strips types but does not transform JSX. */

export type ExactConcurrencyState = "on" | "off" | "unavailable";

/** `off` for anything unrecognised, including `undefined` from an older backend:
 *  no answer is no guarantee. */
export function normalizeExactConcurrency(
  value: string | null | undefined,
): ExactConcurrencyState {
  return value === "on" || value === "unavailable" ? value : "off";
}

const EXACT_MEANING =
  "identical output regardless of other chats sharing this model";

const RECOMPUTED_NOTE =
  "This answer was re-prefilled after a park the server could not hold, so it is not guaranteed byte-identical.";

/** `off` renders nothing. `recomputed` is per thread, since the server reports it per answer. */
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
      // Not "refused": Studio withholds the mode itself on some launches.
      title: `Exact concurrency was requested (${EXACT_MEANING}), but this load is not running with it, so a chat's output can depend on the other chats sharing this model.${note}`,
    };
  }
  return null;
}

/** The state is the resident local server's: a hosted model, or another local checkpoint picked
 *  while the previous one is still loaded, is not what it describes. */
export function exactConcurrencyChipApplies({
  isExternalModel,
  residentCheckpoint,
  selectedCheckpoint,
  modelLoading = false,
}: {
  isExternalModel: boolean;
  residentCheckpoint: string | null | undefined;
  selectedCheckpoint: string | null | undefined;
  modelLoading?: boolean;
}): boolean {
  return (
    !isExternalModel &&
    !modelLoading &&
    residentCheckpoint != null &&
    selectedCheckpoint === residentCheckpoint
  );
}
