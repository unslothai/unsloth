// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ThreadRecord } from "../types";

export type CompareVariant = "general" | "lora";
export type CompareThreadShape = CompareVariant;

export type ComparePaneThreadIds = {
  shape: CompareThreadShape | null;
  first: string | undefined;
  second: string | undefined;
};

function threadId(
  threads: ThreadRecord[],
  modelType: string,
): string | undefined {
  return threads.find((thread) => thread.modelType === modelType)?.id;
}

/** Pick one persisted shape for both panes so interrupted writes cannot splice pairs. */
export function resolveComparePaneThreadIds(
  threads: ThreadRecord[],
): ComparePaneThreadIds {
  const model1 = threadId(threads, "model1");
  const model2 = threadId(threads, "model2");
  const base = threadId(threads, "base");
  const lora = threadId(threads, "lora");

  if ((model1 && model2) || ((model1 || model2) && !(base && lora))) {
    return { shape: "general", first: model1, second: model2 };
  }
  if (base || lora) {
    return { shape: "lora", first: base, second: lora };
  }
  return { shape: null, first: undefined, second: undefined };
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
  threads: ThreadRecord[],
  checkpointIsLora: boolean,
): CompareVariant {
  const { shape } = resolveComparePaneThreadIds(threads);
  return shape !== "general" && checkpointIsLora ? "lora" : "general";
}
