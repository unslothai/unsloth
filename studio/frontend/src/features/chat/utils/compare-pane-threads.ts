// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ModelType, ThreadRecord } from "../types";

export type CompareVariant = "general" | "lora";

/** A pair the generalized compare saved, whatever else it also holds. */
export function pairHasGeneralThreads(threads: ThreadRecord[]): boolean {
  return threads.some(
    (t) => t.modelType === "model1" || t.modelType === "model2",
  );
}

/**
 * The loaded checkpoint decides, as it always has, except that a generalized pair
 * is pinned to the generalized path: handing one to the adapter-toggle path
 * relabels its panes and appends adapter-off and adapter-on answers to the two
 * histories it already holds (#9823).
 *
 * Not pinned the other way. A base/lora pair reopened with a plain checkpoint
 * loaded stays generalized, where the panes carry model selectors; the adapter
 * toggle would warn "model is not a PeftModel" and answer both panes identically.
 */
export function compareVariantForPair(
  isGeneralPair: boolean,
  checkpointIsLora: boolean,
): CompareVariant {
  return !isGeneralPair && checkpointIsLora ? "lora" : "general";
}

/**
 * The generalized panes still recover a base/lora pair reopened with a plain
 * checkpoint loaded (#5910), hence the alias. Two lookups, not one predicate over
 * both: `listStoredChatThreads` sorts updatedAt-descending, so a single find
 * returns the freshest row and can pair the panes across two comparison runs.
 */
export function resolveComparePaneThreadId(
  threads: ThreadRecord[],
  native: ModelType,
  alias: ModelType,
): string | undefined {
  return (
    threads.find((t) => t.modelType === native)?.id ??
    threads.find((t) => t.modelType === alias)?.id
  );
}
