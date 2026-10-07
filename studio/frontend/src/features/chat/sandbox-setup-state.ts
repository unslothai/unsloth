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
  loadFailed: boolean;
  install: "linux" | "windows" | null;
  showConsent: boolean;
  installDisabled: boolean;
  command: string;
  showOwnerOnly: boolean;
  showRunInTerminal: boolean;
  running: boolean;
  result: JobResult;
  outputLines: string[];
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
  const showConsent = install === "windows" && capability.needsConsent;
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
      (command !== "" ||
        capability.setupBlocked === "not_owner" ||
        capability.setupBlocked === "not_local") &&
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
