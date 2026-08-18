// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type StoredGenerationStatus =
  | "queued"
  | "running"
  | "cancelling"
  | "cancelled"
  | "completed"
  | "failed";

const TERMINAL = new Set<StoredGenerationStatus>([
  "cancelled",
  "completed",
  "failed",
]);

export function generationNeedsRecovery(
  metadata: Record<string, unknown>,
): boolean {
  const status = String(metadata.generationStatus) as StoredGenerationStatus;
  return (
    typeof metadata.generationRunId === "string" &&
    (metadata.generationSettled !== true || !TERMINAL.has(status))
  );
}

export function generationRecoveryMetadata(options: {
  current: Record<string, unknown>;
  runId: string;
  status: StoredGenerationStatus;
  cursor: number;
  lastEventSeq: number;
  lengthLimited: boolean;
}): Record<string, unknown> {
  const { current, runId, status, cursor, lastEventSeq, lengthLimited } =
    options;
  const settled = TERMINAL.has(status) && cursor >= lastEventSeq;
  const next: Record<string, unknown> = {
    ...current,
    generationRunId: runId,
    generationSeq: cursor,
    generationStatus: status,
    generationSettled: settled,
    serverManaged: true,
  };
  if (status === "completed" && settled) {
    if (lengthLimited) {
      next.incomplete = { reason: "length" };
    } else {
      next.incomplete = undefined;
    }
  } else if (status === "failed") {
    next.incomplete = { reason: "interrupted" };
  } else {
    next.incomplete = { reason: "cancelled" };
  }
  return next;
}

export function shouldPreserveGenerationMetadata(
  existing: Record<string, unknown> | undefined,
  incoming: Record<string, unknown> | undefined,
): boolean {
  if (typeof existing?.generationRunId !== "string") {
    return false;
  }
  const sameRun = existing.generationRunId === incoming?.generationRunId;
  const existingSeq = Number(existing.generationSeq ?? -1);
  const incomingSeq = Number(incoming?.generationSeq ?? -1);
  const existingStatus = String(existing.generationStatus);
  return (
    !sameRun ||
    incoming?.serverManaged !== true ||
    existingSeq > incomingSeq ||
    (TERMINAL.has(existingStatus as StoredGenerationStatus) &&
      incoming?.generationStatus !== existing.generationStatus) ||
    (existing.generationSettled === true &&
      incoming?.generationSettled !== true)
  );
}

type RecoveryEventTarget = Pick<
  EventTarget,
  "addEventListener" | "removeEventListener"
>;
type RecoveryVisibilityTarget = RecoveryEventTarget & {
  readonly visibilityState: string;
};

export function subscribeGenerationRecoveryTriggers(
  windowTarget: RecoveryEventTarget,
  documentTarget: RecoveryVisibilityTarget,
  recover: () => void,
): () => void {
  const onVisible = () => {
    if (documentTarget.visibilityState === "visible") {
      recover();
    }
  };
  windowTarget.addEventListener("online", recover);
  windowTarget.addEventListener("pageshow", recover);
  documentTarget.addEventListener("visibilitychange", onVisible);
  return () => {
    windowTarget.removeEventListener("online", recover);
    windowTarget.removeEventListener("pageshow", recover);
    documentTarget.removeEventListener("visibilitychange", onVisible);
  };
}
