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

/** Persisted shape owns the renderer; reclassifying from the checkpoint relabels histories. */
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
  isExternal: boolean;
  /** `residentCheckpoint === undefined`: no status read has landed yet. */
  residentUnknown: boolean;
  models: readonly { id: string; isLora?: boolean }[];
  loras: readonly { id: string; exportType?: string }[];
  inventorySettled: boolean;
};

/** Whether the loaded checkpoint is a LoRA; `null` only while genuinely unclassified. */
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
  // An explicit catalog row answers alone; waiting on the deferred inventory blanked the pair.
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

/** Every outcome reaches a rendered state: a failure retries once, then shows a visible surface. */
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
