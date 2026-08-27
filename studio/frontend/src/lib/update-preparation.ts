// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type ShellPreparation = "pending" | "downloading" | "done" | "failed";

export type BackendPreparation =
  | "pending"
  | "waiting"
  | "staging"
  | "ready"
  | "failed"
  | "skipped";

export interface UpdatePreparation {
  shell: ShellPreparation;
  backend: BackendPreparation;
  shellProgress: number;
}

export type PreparationStatus = "preparing" | "ready" | "available";

export type RestartPlan = "fast" | "classic";

export interface StagedUpdateStatus {
  state: "none" | "partial" | "ready" | "failed";
  backendVersion: string | null;
  shellVersion: string | null;
}

export type StagingDecision = "stage" | "already-ready" | "skip";

export const INITIAL_PREPARATION: UpdatePreparation = {
  shell: "pending",
  backend: "pending",
  shellProgress: 0,
};

const LEADING_V = /^v/;

function sameVersion(left: string | null | undefined, right: string): boolean {
  if (!left) return false;
  return left.replace(LEADING_V, "") === right.replace(LEADING_V, "");
}

export function preparationStatus(preparation: UpdatePreparation): PreparationStatus {
  if (preparation.shell === "failed") return "available";
  if (preparation.shell !== "done") return "preparing";
  switch (preparation.backend) {
    case "ready":
    case "failed":
    case "skipped":
      return "ready";
    default:
      return "preparing";
  }
}

export function restartPlan(preparation: UpdatePreparation): RestartPlan {
  return preparation.shell === "done" && preparation.backend === "ready"
    ? "fast"
    : "classic";
}

export function stagingDecision(args: {
  inApp: boolean;
  isExternalServer: boolean;
  offeredVersion: string;
  staged: StagedUpdateStatus;
}): StagingDecision {
  if (!args.inApp || args.isExternalServer) return "skip";
  const matches = sameVersion(args.staged.shellVersion, args.offeredVersion);
  if (args.staged.state === "ready" && matches) return "already-ready";
  if (args.staged.state === "failed" && matches) return "skip";
  return "stage";
}

export function backendIdle(
  health: { inference_active?: boolean } | null,
  trainingActive: boolean,
): boolean {
  if (trainingActive) return false;
  return health !== null && health.inference_active !== true;
}

export function downloadPercent(downloaded: number, total: number | null): number {
  if (!total || total <= 0) return 0;
  return Math.min(100, Math.round((downloaded / total) * 100));
}

export function preparationShortLabel(preparation: UpdatePreparation): string {
  if (preparation.shell === "downloading") return `downloading ${preparation.shellProgress}%`;
  switch (preparation.backend) {
    case "waiting":
      return "waiting for idle";
    case "staging":
      return "setting up backend";
    default:
      return "starting";
  }
}

export const IDLE_POLL_MS = 20_000;
// A backend that is stopped, crashed, or will not start never reports idle, and
// the update offer is the user's way out of exactly that. Waiting forever leaves
// the pill with no action and the settings button disabled, so the wait is bounded
// and the offer falls back to the classic update.
export const IDLE_WAIT_MS = 5 * 60_000;

export type IdleWaitOutcome = "idle" | "timeout" | "cancelled";

export async function waitForBackendIdle(deps: {
  cancelled: () => boolean;
  probe: () => Promise<boolean>;
  sleep: (ms: number) => Promise<void>;
  now: () => number;
  pollMs?: number;
  timeoutMs?: number;
}): Promise<IdleWaitOutcome> {
  const pollMs = deps.pollMs ?? IDLE_POLL_MS;
  const deadline = deps.now() + (deps.timeoutMs ?? IDLE_WAIT_MS);
  while (!deps.cancelled()) {
    if (await deps.probe()) return "idle";
    if (deps.cancelled()) return "cancelled";
    if (deps.now() >= deadline) return "timeout";
    await deps.sleep(pollMs);
  }
  return "cancelled";
}
