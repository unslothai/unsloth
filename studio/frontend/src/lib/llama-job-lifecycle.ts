// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type LlamaJobState = "idle" | "running" | "success" | "error";
export type LlamaJobOperation = "update" | "switch" | null;

interface LlamaJob {
  state: LlamaJobState;
  operation: LlamaJobOperation;
}

interface TimestampedLlamaJob extends LlamaJob {
  started_at: string | null;
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

export type UpdateComponent = "llama.cpp" | "whisper.cpp";

export interface UpdateComponentFlags {
  llama: boolean;
  whisper: boolean;
}

/**
 * Which component's offer the single update card shows.
 *
 * Both can be pending at once and the backend names only one: llama.cpp when its
 * release is behind, whisper.cpp otherwise, so a llama.cpp backend migration is
 * named whisper.cpp when whisper is stale too. If the named one's switch is off
 * and the other has an offer the user does allow, the card shows that one rather
 * than nothing. Update installs everything pending either way.
 */
export function updateBannerComponent(
  named: UpdateComponent,
  pending: UpdateComponentFlags,
  allow: UpdateComponentFlags,
): UpdateComponent {
  const key = named === "whisper.cpp" ? "whisper" : "llama";
  if (allow[key]) {
    return named;
  }
  const other = key === "whisper" ? "llama" : "whisper";
  return allow[other] && pending[other]
    ? ((other === "whisper" ? "whisper.cpp" : "llama.cpp") as UpdateComponent)
    : named;
}

/**
 * The tag a finished update reports.
 *
 * The job's `to_tag` is the llama.cpp build by definition, so a card showing the
 * whisper.cpp offer reports the release it advertised instead of llama's.
 */
export function updateToastTag(
  component: UpdateComponent,
  jobTag: string | null | undefined,
  offerTag: string | null | undefined,
): string | null {
  const preferred = component === "whisper.cpp" ? offerTag : jobTag;
  return preferred ?? offerTag ?? jobTag ?? null;
}

/**
 * Which notification switch answers for the update card, held across a job.
 *
 * The card is muted by the component it names, and a chained apply renames it
 * when the llama.cpp phase lands and the whisper.cpp phase starts. Reading the
 * live switch there would take a running update off screen halfway through, so
 * the switch the card started under is held until the job is over. `null` means
 * nothing is held and the live switch applies.
 */
export function heldUpdateBannerPref(
  held: boolean | null,
  inFlight: boolean,
  live: boolean,
): boolean | null {
  return inFlight ? (held ?? live) : null;
}

/**
 * Whether the banner's version line has anything to say.
 *
 * `updateAvailable` is the only field reporting that the release moved. The tags cannot:
 * `installed_tag` is normalized (`b9596`) while `latest_tag` is the full identity
 * (`b9596-mix-<sha>`), so a fork install shows them differing at the release it is
 * running -- which is exactly where a migration is offered.
 */
/**
 * A poll that left while the job was still running can return after a later
 * poll already observed success or error. Adopting that running snapshot
 * re-pins the "Updating..." toast after the completion poller has stopped.
 */
export function llamaUpdateSnapshotIsStale(
  adopted: TimestampedLlamaJob,
  incoming: TimestampedLlamaJob,
): boolean {
  if (adopted.state !== "success" && adopted.state !== "error") {
    return false;
  }
  return (
    incoming.state === "running" &&
    adopted.started_at != null &&
    adopted.started_at === incoming.started_at
  );
}

export function llamaReleaseChanged(
  updateAvailable: boolean,
  installedTag: string | null,
  latestTag: string | null,
): boolean {
  return Boolean(
    updateAvailable && installedTag && latestTag && installedTag !== latestTag,
  );
}

/** What to tell the user a finished Update actually did.
 *
 * A migration runs at the release already installed and can end on the backend already
 * installed, so "updated to <tag>" describes neither -- and the tag is llama's even when
 * a pending whisper update named the toast. The job's own message is accurate.
 */
export function llamaUpdateToastMessage({
  component,
  migrating,
  jobMessage,
  updatedTag,
  reloadRequired,
}: {
  component: string;
  migrating: boolean;
  jobMessage: string | null | undefined;
  updatedTag: string;
  reloadRequired: boolean | null | undefined;
}): string {
  const reloadHint = reloadRequired ? " Reload your model to use it." : "";
  const migrationMessage = migrating ? (jobMessage ?? "").trim() : "";
  if (!migrationMessage) {
    return `${component} updated to ${updatedTag}.${reloadHint}`;
  }
  // The phase appends its own reload hint when it has one to give.
  return migrationMessage.includes("Reload")
    ? migrationMessage
    : `${migrationMessage}${reloadHint}`;
}
