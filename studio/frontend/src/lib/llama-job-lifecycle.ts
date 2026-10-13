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

export type UpdateComponent = "llama.cpp" | "whisper.cpp" | "audio.cpp";

export interface UpdateComponentFlags {
  llama: boolean;
  whisper: boolean;
  audio: boolean;
}

// backend naming order matches the job installation order.
const UPDATE_COMPONENT_KEYS: Record<
  UpdateComponent,
  keyof UpdateComponentFlags
> = {
  "llama.cpp": "llama",
  "whisper.cpp": "whisper",
  "audio.cpp": "audio",
};

export function updateComponentKey(
  component: UpdateComponent,
): keyof UpdateComponentFlags {
  return UPDATE_COMPONENT_KEYS[component];
}

/** pending components remain listed even when their notifications are muted. */
export function updateBannerComponents(
  pending: UpdateComponentFlags,
): UpdateComponent[] {
  return (Object.keys(UPDATE_COMPONENT_KEYS) as UpdateComponent[]).filter(
    (component) => pending[UPDATE_COMPONENT_KEYS[component]],
  );
}

export function audibleComponents(
  listed: readonly UpdateComponent[],
  muted: UpdateComponentFlags,
): UpdateComponent[] {
  return listed.filter((component) => !muted[UPDATE_COMPONENT_KEYS[component]]);
}

/** an audible due component opens the card; it stays open until every listed component is muted. */
export function updateBannerAllowed(
  listed: readonly UpdateComponent[],
  due: UpdateComponentFlags,
  muted: UpdateComponentFlags,
  open: boolean,
): boolean {
  const audible = audibleComponents(listed, muted);
  return (
    audible.length > 0 &&
    (open || audible.some((component) => due[UPDATE_COMPONENT_KEYS[component]]))
  );
}

/** holds initial rows as phases leave status; releasing the hold lets retries use remaining rows. */
export function heldJobRows<T>(
  held: T[] | null,
  applying: boolean,
  live: T[],
): T[] | null {
  if (!applying) {
    return null;
  }
  return held ?? (live.length > 0 ? live : null);
}

export function updateComponentsLabel(
  components: readonly UpdateComponent[],
): string {
  if (components.length <= 1) {
    return components[0] ?? "llama.cpp";
  }
  return `${components.slice(0, -1).join(", ")} and ${components[components.length - 1]}`;
}

/** non-llama offers use their advertised tag because job `to_tag` always identifies llama.cpp. */
export function updateToastTag(
  component: UpdateComponent,
  jobTag: string | null | undefined,
  offerTag: string | null | undefined,
): string | null {
  const preferred = component === "llama.cpp" ? jobTag : offerTag;
  return preferred ?? offerTag ?? jobTag ?? null;
}

/** holds the opening preference because completed phases disappear during a chained job. */
export function heldUpdateBannerPref(
  held: boolean | null,
  inFlight: boolean,
  live: boolean,
): boolean | null {
  return inFlight ? (held ?? live) : null;
}

/** trust `updateAvailable` because installed tags are normalized but fork identities are not. */
export function llamaReleaseChanged(
  updateAvailable: boolean,
  installedTag: string | null,
  latestTag: string | null,
): boolean {
  return Boolean(
    updateAvailable && installedTag && latestTag && installedTag !== latestTag,
  );
}

/** migration messages avoid misleading tags; null tags omit versions for bundled updates. */
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
  updatedTag: string | null;
  reloadRequired: boolean | null | undefined;
}): string {
  const reloadHint = reloadRequired ? " Reload your model to use it." : "";
  const migrationMessage = migrating ? (jobMessage ?? "").trim() : "";
  if (!migrationMessage) {
    const toTag = updatedTag === null ? "" : ` to ${updatedTag}`;
    return `${component} updated${toTag}.${reloadHint}`;
  }
  // avoid duplicating a reload hint already supplied by the phase.
  return migrationMessage.includes("Reload")
    ? migrationMessage
    : `${migrationMessage}${reloadHint}`;
}
