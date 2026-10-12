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

// In the order the backend names them, which is also the order the job installs them.
const UPDATE_COMPONENT_KEYS: Record<
  UpdateComponent,
  keyof UpdateComponentFlags
> = {
  "llama.cpp": "llama",
  "whisper.cpp": "whisper",
  "audio.cpp": "audio",
};

/** The notification channel and status field that answer for a component. */
export function updateComponentKey(
  component: UpdateComponent,
): keyof UpdateComponentFlags {
  return UPDATE_COMPONENT_KEYS[component];
}

/**
 * Which components the single update card lists, in install order.
 *
 * Update installs everything pending in one job, so one appearance names every
 * pending component. A component whose notifications are off is still installed,
 * only not listed.
 */
export function updateBannerComponents(
  pending: UpdateComponentFlags,
  muted: UpdateComponentFlags,
): UpdateComponent[] {
  return (Object.keys(UPDATE_COMPONENT_KEYS) as UpdateComponent[]).filter(
    (component) =>
      pending[UPDATE_COMPONENT_KEYS[component]] &&
      !muted[UPDATE_COMPONENT_KEYS[component]],
  );
}

/**
 * Whether the update card may appear.
 *
 * One appearance covers every listed component, so it opens when any of them is due
 * and stays allowed once open, after it has recorded them all as shown.
 */
export function updateBannerAllowed(
  listed: readonly UpdateComponent[],
  due: UpdateComponentFlags,
  open: boolean,
): boolean {
  return (
    listed.length > 0 &&
    (open || listed.some((component) => due[UPDATE_COMPONENT_KEYS[component]]))
  );
}

/** "llama.cpp", "llama.cpp and whisper.cpp", "llama.cpp, whisper.cpp and audio.cpp". */
export function updateComponentsLabel(
  components: readonly UpdateComponent[],
): string {
  if (components.length <= 1) {
    return components[0] ?? "llama.cpp";
  }
  return `${components.slice(0, -1).join(", ")} and ${components[components.length - 1]}`;
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
 * Which notification switch answers for the update card, held across a job.
 *
 * The card is allowed by the components it lists, and a chained apply drops each
 * one from the status as its phase lands, leaving only components that may be
 * muted. Reading the live switch there would take a running update off screen
 * halfway through, so the switch the card started under is held until the job is
 * over. `null` means nothing is held and the live switch applies.
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
