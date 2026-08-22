// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type LlamaJobState = "idle" | "running" | "success" | "error";
export type LlamaJobOperation = "update" | "switch" | null;

interface LlamaJob {
  state: LlamaJobState;
  operation: LlamaJobOperation;
}

interface IdentifiedLlamaJob extends LlamaJob {
  startedAt: string | null;
}

export type OwnedLlamaSwitchOutcome =
  | "running"
  | "success"
  | "error"
  | "interrupted";

/** Interpret a status only as the switch job that this surface accepted. */
export function ownedLlamaSwitchOutcome(
  job: IdentifiedLlamaJob,
  acceptedStartedAt: string | null,
): OwnedLlamaSwitchOutcome {
  if (
    !acceptedStartedAt ||
    job.startedAt !== acceptedStartedAt ||
    job.operation !== "switch"
  ) {
    return "interrupted";
  }
  return job.state === "idle" ? "interrupted" : job.state;
}

/**
 * Whether an `already_running` /update response is the update this apply asked
 * for. A backend switch shares the same job: adopting it would resolve this
 * action as an applied update while the pending release is still uninstalled.
 */
export function llamaUpdateAdoptsRunningJob(
  reason: string | null | undefined,
  job: LlamaJob,
): boolean {
  return reason === "already_running" && job.operation !== "switch";
}

export interface LlamaUpdatePresentation {
  applying: boolean;
  visible: boolean;
  running: boolean;
}

/** Derive the update banner from every shared-job status transition. */
export function llamaUpdatePresentation(
  updateAvailable: boolean,
  job: LlamaJob,
): LlamaUpdatePresentation {
  if (job.state !== "running") {
    return { applying: false, visible: updateAvailable, running: false };
  }
  const switching = job.operation === "switch";
  return {
    applying: !switching,
    visible: !switching,
    running: true,
  };
}

export const LLAMA_JOB_STATUS_TIMEOUT_MS = 15_000;

export function llamaStatusRequestIsStale(
  latestAppliedRequest: number,
  incomingRequest: number,
): boolean {
  return incomingRequest < latestAppliedRequest;
}

export async function boundedLlamaStatusRequest<T>(
  request: (signal: AbortSignal) => Promise<T>,
  timeoutMs = LLAMA_JOB_STATUS_TIMEOUT_MS,
): Promise<T | null> {
  const controller = new AbortController();
  let timeout: ReturnType<typeof globalThis.setTimeout> | undefined;
  const deadline = new Promise<null>((resolve) => {
    timeout = globalThis.setTimeout(() => {
      controller.abort();
      resolve(null);
    }, timeoutMs);
  });
  try {
    return await Promise.race([
      request(controller.signal).catch(() => null),
      deadline,
    ]);
  } finally {
    if (timeout !== undefined) {
      globalThis.clearTimeout(timeout);
    }
  }
}
