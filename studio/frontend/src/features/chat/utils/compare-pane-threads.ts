// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { modelIdsMatch } from "../../hub/lib/model-identity.ts";
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

/** A persisted shape owns the renderer. Reclassifying it from the loaded checkpoint relabels
 *  existing histories and can write the other comparison mode into them. */
export function compareVariantForPair(
  threads: ThreadRecord[],
  checkpointIsLora: boolean | null,
): CompareVariant | null {
  const { shape } = resolveComparePaneThreadIds(threads);
  if (shape) return shape;
  if (checkpointIsLora === null) return null;
  return checkpointIsLora ? "lora" : "general";
}

export type CheckpointCompareClassInput = {
  checkpoint: string | null | undefined;
  /** External providers have no local adapter, so they are never a LoRA compare. */
  isExternal: boolean;
  /** `residentCheckpoint === undefined`: no status read has landed yet. */
  residentUnknown: boolean;
  models: readonly { id: string; isLora?: boolean }[];
  loras: readonly { id: string; exportType?: string }[];
  inventorySettled: boolean;
};

/** Is the loaded checkpoint a LoRA, meaning a base-vs-fine-tuned compare on the fast simultaneous
 *  adapter-toggle path? `null` only while it is genuinely unclassified, since that blanks the
 *  compare view. */
export function checkpointCompareClass(
  input: CheckpointCompareClassInput,
): boolean | null {
  if (input.isExternal) return false;
  if (input.residentUnknown && !input.inventorySettled) return null;
  const checkpoint = input.checkpoint;
  if (!checkpoint) return false;
  const row = input.models.find((model) => modelIdsMatch(model.id, checkpoint));
  if (row?.isLora) return true;
  if (
    input.loras.find((lora) => modelIdsMatch(lora.id, checkpoint))
      ?.exportType === "lora"
  ) {
    return true;
  }
  if (input.inventorySettled) return false;
  // An explicit catalog row answers on its own. The deferred inventory could only add an adapter
  // row, which this one rules out, and chat defers it by 1.2s, so waiting rendered a new pair
  // blank for at least that long.
  return row ? false : null;
}

export type ComparePairReadOutcome =
  | { threads: ThreadRecord[] }
  | { failed: true };

export type ComparePairReadState =
  | { status: "pending" }
  | { status: "retry" }
  | { status: "unreadable" }
  | { status: "ready"; variant: CompareVariant };

/** Every outcome of the pair read reaches a rendered state. A failure retries once and then asks
 *  for a visible surface: leaving it unsettled renders nothing at all, and settling it as an
 *  empty pair picks a renderer for a pair whose shape is unknown. */
export function comparePairReadState(
  outcome: ComparePairReadOutcome,
  checkpointIsLora: boolean | null,
  attempt: number,
): ComparePairReadState {
  if ("failed" in outcome) {
    return attempt === 0 ? { status: "retry" } : { status: "unreadable" };
  }
  const variant = compareVariantForPair(outcome.threads, checkpointIsLora);
  return variant === null
    ? { status: "pending" }
    : { status: "ready", variant };
}
