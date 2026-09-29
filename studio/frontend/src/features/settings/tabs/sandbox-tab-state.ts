// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  HostPrepJob,
  SandboxToolStatus,
  TerminalShell,
  WindowsSandboxStatus,
} from "../api/sandbox-isolation";

export const HOST_PREP_POLL_MS = 2000;

// The one step wxc-host-prep undoes on every restart; missing alone, the PC was prepared before.
const REPEATING_STEP = "prepare-null-device";

const BACKEND_LABELS: Record<string, string> = {
  bubblewrap: "bubblewrap",
  "macos-seatbelt": "Seatbelt",
  "mxc-processcontainer": "MXC",
};

export type ToolRowView = {
  isolated: boolean;
  backendLabel: string;
  reason: string;
  runsInCmd: boolean;
};

export function toolRowView(
  tool: SandboxToolStatus,
  shell: TerminalShell | null = null,
): ToolRowView {
  return {
    isolated: tool.available,
    backendLabel: BACKEND_LABELS[tool.backend] ?? tool.backend,
    reason: tool.reason,
    runsInCmd: shell === "cmd_isolated",
  };
}

export type HostPrepStatus =
  | "runtimeMissing"
  | "off"
  | "prepared"
  | "needsPreparing"
  | "needsPreparingAgain"
  | "unknown";

export type WindowsView = {
  runtimeMissing: boolean;
  optInChecked: boolean;
  optInDisabled: boolean;
  optInLocked: boolean;
  showGrantsRow: boolean;
  grantsChecked: boolean;
  grantsDisabled: boolean;
  grantsLocked: boolean;
  prep: HostPrepStatus;
  showPrepareButton: boolean;
  prepareDisabled: boolean;
};

export function hostPrepStatus(windows: WindowsSandboxStatus): HostPrepStatus {
  if (!windows.runtimeInstalled) return "runtimeMissing";
  if (!windows.allowDaclFallback) return "off";
  const missing = windows.hostPrepMissing;
  if (missing === null) return "unknown";
  if (missing.length === 0) return "prepared";
  if (missing.length === 1 && missing[0] === REPEATING_STEP) {
    return "needsPreparingAgain";
  }
  return "needsPreparing";
}

export function windowsView(
  windows: WindowsSandboxStatus,
  job: HostPrepJob | null,
  saving: boolean,
): WindowsView {
  const prep = hostPrepStatus(windows);
  const running = job?.state === "running";
  const runtimeMissing = prep === "runtimeMissing";
  return {
    runtimeMissing,
    optInChecked: windows.allowDaclFallback,
    optInDisabled:
      runtimeMissing || windows.daclLockedByEnvironment || saving || running,
    optInLocked: windows.daclLockedByEnvironment,
    showGrantsRow: !runtimeMissing && windows.allowDaclFallback,
    grantsChecked: windows.persistentReadGrants,
    grantsDisabled: windows.grantsLockedByEnvironment || saving || running,
    grantsLocked: windows.grantsLockedByEnvironment,
    prep,
    // A running job keeps the button so its progress stays where the owner clicked.
    showPrepareButton:
      running ||
      prep === "needsPreparing" ||
      prep === "needsPreparingAgain" ||
      prep === "unknown",
    prepareDisabled: running || saving,
  };
}

export type JobResult = "succeeded" | "declined" | "failed" | null;

export function jobResult(job: HostPrepJob | null): JobResult {
  if (!job) return null;
  if (
    job.state === "succeeded" ||
    job.state === "declined" ||
    job.state === "failed"
  ) {
    return job.state;
  }
  return null;
}

export function shouldPollJob(job: HostPrepJob | null): boolean {
  return job?.state === "running";
}

// Only a failure shows the helper's output; the last few lines are the ones that name the cause.
export function jobOutputLines(job: HostPrepJob | null, max = 6): string[] {
  if (!job || job.state !== "failed") return [];
  return job.outputTail.filter((line) => line.trim() !== "").slice(-max);
}
