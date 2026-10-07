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

export type UpdateComponent = "llama.cpp" | "whisper.cpp";

export interface UpdateComponentFlags {
  llama: boolean;
  whisper: boolean;
}

/**
 * The backend names llama.cpp when its release is behind, else whisper.cpp. If the named
 * component's switch is off, fall back to the other allowed offer.
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

/** `to_tag` is always llama.cpp's, so a whisper.cpp card reports its advertised release. */
export function updateToastTag(
  component: UpdateComponent,
  jobTag: string | null | undefined,
  offerTag: string | null | undefined,
): string | null {
  const preferred = component === "whisper.cpp" ? offerTag : jobTag;
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
