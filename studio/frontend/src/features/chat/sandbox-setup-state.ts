// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type JobResult, jobOutputLines, jobResult } from "@/features/settings";
import type {
  SandboxCapability,
  SandboxSetupJob,
} from "./api/sandbox-capability";

export const SANDBOX_SETUP_POLL_MS = 2000;

export type SandboxSetupView = {
  checking: boolean;
  // The dialog's own check failed: say so and offer Retry, never an endless spinner.
  loadFailed: boolean;
  // "linux": Install sandbox; "windows": Set up Windows sandbox behind the MXC consent.
  install: "linux" | "windows" | null;
  showConsent: boolean;
  installDisabled: boolean;
  command: string;
  // Someone who cannot start the setup (not the owner, or not at the computer running Unsloth)
  // gets the command with a note saying who can.
  showOwnerOnly: boolean;
  // The owner is here but Unsloth cannot ask for the password: run the command in a terminal.
  showRunInTerminal: boolean;
  running: boolean;
  result: JobResult;
  outputLines: string[];
  // The server's explanation for a declined or failed setup, e.g. "a password is required".
  note: string;
};

export function sandboxSetupView({
  capability,
  job,
  consent,
  loadFailed = false,
}: {
  capability: SandboxCapability | null;
  job: SandboxSetupJob | null;
  consent: boolean;
  loadFailed?: boolean;
}): SandboxSetupView {
  const running = job?.state === "running";
  const result = jobResult(job);
  const note =
    (result === "failed" || result === "declined") && job?.note ? job.note : "";
  if (!capability) {
    return {
      checking: !loadFailed,
      loadFailed,
      install: null,
      showConsent: false,
      installDisabled: true,
      command: "",
      showOwnerOnly: false,
      showRunInTerminal: false,
      running,
      result,
      outputLines: jobOutputLines(job),
      note,
    };
  }
  const action = capability.canRunSetup ? capability.setupAction : null;
  const install =
    action === "linux-install"
      ? "linux"
      : action === "windows-setup"
        ? "windows"
        : null;
  // Only while the MXC opt-in is still off: after a reboot only the preparation is missing.
  const showConsent = install === "windows" && capability.needsConsent;
  // A failed job names the command for the step it stopped at; otherwise the server's plan.
  const command =
    (result === "failed" || result === "declined") && job?.manualCommand
      ? job.manualCommand
      : capability.manualCommand;
  return {
    checking: false,
    loadFailed: false,
    install,
    showConsent,
    installDisabled: running || (showConsent && !consent),
    command,
    showOwnerOnly:
      install === null &&
      command !== "" &&
      capability.setupBlocked !== "no_elevation",
    showRunInTerminal:
      install === null &&
      command !== "" &&
      capability.setupBlocked === "no_elevation",
    running,
    result,
    outputLines: jobOutputLines(job),
    note,
  };
}
