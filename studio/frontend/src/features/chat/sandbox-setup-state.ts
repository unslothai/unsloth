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
  // "linux": Install sandbox; "windows": Set up Windows sandbox behind the MXC consent.
  install: "linux" | "windows" | null;
  showConsent: boolean;
  installDisabled: boolean;
  command: string;
  // Someone who cannot start the setup (not the owner, or not at the computer running Unsloth)
  // gets the command with a note saying who can.
  showOwnerOnly: boolean;
  running: boolean;
  result: JobResult;
  outputLines: string[];
};

export function sandboxSetupView({
  capability,
  job,
  consent,
}: {
  capability: SandboxCapability | null;
  job: SandboxSetupJob | null;
  consent: boolean;
}): SandboxSetupView {
  const running = job?.state === "running";
  const result = jobResult(job);
  if (!capability) {
    return {
      checking: true,
      install: null,
      showConsent: false,
      installDisabled: true,
      command: "",
      showOwnerOnly: false,
      running,
      result,
      outputLines: jobOutputLines(job),
    };
  }
  const action = capability.canRunSetup ? capability.setupAction : null;
  const install =
    action === "linux-install"
      ? "linux"
      : action === "windows-setup"
        ? "windows"
        : null;
  const showConsent = install === "windows";
  // A failed job names the command for the step it stopped at; otherwise the server's plan.
  const command =
    (result === "failed" || result === "declined") && job?.manualCommand
      ? job.manualCommand
      : capability.manualCommand;
  return {
    checking: false,
    install,
    showConsent,
    installDisabled: running || (showConsent && !consent),
    command,
    showOwnerOnly: install === null && command !== "",
    running,
    result,
    outputLines: jobOutputLines(job),
  };
}
