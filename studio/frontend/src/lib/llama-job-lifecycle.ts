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

/** A backend switch shares the update job, so adopting it would mark a pending update applied. */
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

export type UpdateComponent = "llama.cpp" | "whisper.cpp" | "audio.cpp";

export interface UpdateComponentFlags {
  llama: boolean;
  whisper: boolean;
  audio: boolean;
}

// In the order the backend names them, which is also the order the job installs them.
const UPDATE_COMPONENT_KEYS: Record<
  UpdateComponent,
  keyof UpdateComponentFlags
> = {
  "llama.cpp": "llama",
  "whisper.cpp": "whisper",
  "audio.cpp": "audio",
};

/**
 * Which component's offer the single update card shows.
 *
 * Several can be pending at once and the backend names only one: llama.cpp when its
 * release is behind, then whisper.cpp, then audio.cpp, so a llama.cpp backend
 * migration is named whisper.cpp when whisper is stale too. If the named one's
 * switch is off and another has an offer the user does allow, the card shows that
 * one rather than nothing. Update installs everything pending either way.
 */
export function updateBannerComponent(
  named: UpdateComponent,
  pending: UpdateComponentFlags,
  allow: UpdateComponentFlags,
): UpdateComponent {
  if (allow[UPDATE_COMPONENT_KEYS[named]]) {
    return named;
  }
  const other = (Object.keys(UPDATE_COMPONENT_KEYS) as UpdateComponent[]).find(
    (component) =>
      component !== named &&
      allow[UPDATE_COMPONENT_KEYS[component]] &&
      pending[UPDATE_COMPONENT_KEYS[component]],
  );
  return other ?? named;
}

/**
 * The tag a finished update reports.
 *
 * The job's `to_tag` is the llama.cpp build by definition, so a card showing the
 * whisper.cpp or audio.cpp offer reports the release it advertised instead of llama's.
 */
export function updateToastTag(
  component: UpdateComponent,
  jobTag: string | null | undefined,
  offerTag: string | null | undefined,
): string | null {
  const preferred = component === "llama.cpp" ? jobTag : offerTag;
  return preferred ?? offerTag ?? jobTag ?? null;
}

/**
 * A chained apply renames the card mid-job, so the starting switch is held until the job ends.
 * `null` means nothing is held and the live switch applies.
 */
export function heldUpdateBannerPref(
  held: boolean | null,
  inFlight: boolean,
  live: boolean,
): boolean | null {
  return inFlight ? (held ?? live) : null;
}

/** Gated on `updateAvailable`: tags alone cannot tell, as `installed_tag` is normalized. */
export function llamaReleaseChanged(
  updateAvailable: boolean,
  installedTag: string | null,
  latestTag: string | null,
): boolean {
  return Boolean(
    updateAvailable && installedTag && latestTag && installedTag !== latestTag,
  );
}

/** A migration may keep the same tag and backend, so the job's own message is used. */
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
  return migrationMessage.includes("Reload")
    ? migrationMessage
    : `${migrationMessage}${reloadHint}`;
}
