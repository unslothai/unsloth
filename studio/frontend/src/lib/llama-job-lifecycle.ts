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

/**
 * Whether the banner's version line has anything to say.
 *
 * `updateAvailable` is the only field that reports the release moved. The two
 * tags cannot answer it: `installed_tag` is deliberately the normalized base tag
 * (`b9596`) while `latest_tag` is the full release identity (`b9596-mix-<sha>`),
 * so on a fork install they differ at the very release the machine is running.
 * A backend migration is offered exactly there, and comparing the tags alone
 * would announce a version update that does not exist and open a changelog
 * between a tag and itself.
 */
export function llamaReleaseChanged(
  updateAvailable: boolean,
  installedTag: string | null,
  latestTag: string | null,
): boolean {
  return Boolean(
    updateAvailable && installedTag && latestTag && installedTag !== latestTag,
  );
}
