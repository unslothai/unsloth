// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type ExactConcurrencyState = "on" | "off" | "unavailable";

export function normalizeExactConcurrency(
  value: string | null | undefined,
): ExactConcurrencyState {
  return value === "on" || value === "unavailable" ? value : "off";
}

const EXACT_MEANING =
  "identical output regardless of other chats sharing this model";

const RECOMPUTED_NOTE =
  "This answer was re-prefilled after a park the server could not hold, so it is not guaranteed byte-identical.";

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
      title: `Exact concurrency was requested (${EXACT_MEANING}), but this load is not running with it, so a chat's output can depend on the other chats sharing this model.${note}`,
    };
  }
  return null;
}

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
