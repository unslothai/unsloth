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
 * Update installs everything pending in one job, so the card names all of it,
 * including components whose notifications are off.
 */
export function updateBannerComponents(
  pending: UpdateComponentFlags,
): UpdateComponent[] {
  return (Object.keys(UPDATE_COMPONENT_KEYS) as UpdateComponent[]).filter(
    (component) => pending[UPDATE_COMPONENT_KEYS[component]],
  );
}

/** The listed components whose notifications are on: they open the card and are recorded. */
export function audibleComponents(
  listed: readonly UpdateComponent[],
  muted: UpdateComponentFlags,
): UpdateComponent[] {
  return listed.filter((component) => !muted[UPDATE_COMPONENT_KEYS[component]]);
}

/**
 * Whether the update card may appear.
 *
 * Only a component whose notifications are on can open it, when it is due. Once open
 * it stays, until every listed component has its notifications turned off.
 */
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

/**
 * The rows a running job keeps on its card.
 *
 * A chained apply clears each component from the status as its phase lands, so the
 * card holds the rows it started with while the job runs. Only while it runs: a failed
 * phase hands back the live rows, so a retry names only what is still pending.
 */
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
 * A `null` tag is for a toast naming several components, where no one tag fits.
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
  updatedTag: string | null;
  reloadRequired: boolean | null | undefined;
}): string {
  const reloadHint = reloadRequired ? " Reload your model to use it." : "";
  const migrationMessage = migrating ? (jobMessage ?? "").trim() : "";
  if (!migrationMessage) {
    const toTag = updatedTag === null ? "" : ` to ${updatedTag}`;
    return `${component} updated${toTag}.${reloadHint}`;
  }
  // The phase appends its own reload hint when it has one to give.
  return migrationMessage.includes("Reload")
    ? migrationMessage
    : `${migrationMessage}${reloadHint}`;
}
