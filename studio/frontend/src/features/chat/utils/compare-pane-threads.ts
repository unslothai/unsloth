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
 * A persisted shape owns the renderer. Reclassifying it from the loaded checkpoint
 * relabels existing histories and can write the other comparison mode into them.
 */
export function compareVariantForPair(
  threads: ThreadRecord[],
  checkpointIsLora: boolean | null,
): CompareVariant | null {
  const { shape } = resolveComparePaneThreadIds(threads);
  if (shape) return shape;
  if (checkpointIsLora === null) return null;
  return checkpointIsLora ? "lora" : "general";
}
