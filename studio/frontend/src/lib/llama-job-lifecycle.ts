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

/** How many consecutive poll fetches may return nothing before the run is
 * treated as stalled (500 ms apart, so 30 ≈ 15 s of dead fetches). */
export const LLAMA_JOB_POLL_MISS_LIMIT = 30;

export type LlamaJobPollTick =
  | { kind: "polling" }
  | { kind: "finished"; state: "success" | "error" | "idle" }
  | { kind: "stalled" };

/** One job-poll tick's decision, from the fetched status and the streak of
 * fetches that returned nothing.
 *
 * The streak keeps the "applying" flag honest: the poll owns that flag for the
 * whole run, so when every fetch starts failing — an auth token that expired,
 * a network drop, a backend restart window — a tick that only skips leaves the
 * update toast pinned on "Updating..." forever, with its close affordance
 * hidden (#9196). Past the limit the tick reports `stalled`, and the caller
 * drops applying and re-checks once instead of waiting on a backend that is
 * not answering. A fetch that succeeds resets the streak, so a healthy run
 * with occasional timeouts never stalls. */
export function llamaJobPollTick(
  status: { update_available: boolean; job: LlamaJob } | null,
  consecutiveMisses: number,
): LlamaJobPollTick {
  if (status === null) {
    return consecutiveMisses >= LLAMA_JOB_POLL_MISS_LIMIT
      ? { kind: "stalled" }
      : { kind: "polling" };
  }
  if (status.job.state === "running") {
    return { kind: "polling" };
  }
  return { kind: "finished", state: status.job.state };
}
