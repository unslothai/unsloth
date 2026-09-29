// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type {
  HostPrepJob,
  WindowsSandboxStatus,
} from "../src/features/settings/api/sandbox-isolation.ts";
import {
  hostPrepStatus,
  jobOutputLines,
  jobResult,
  shouldPollJob,
  toolRowView,
  windowsView,
} from "../src/features/settings/tabs/sandbox-tab-state.ts";

const windows = (
  overrides: Partial<WindowsSandboxStatus> = {},
): WindowsSandboxStatus => ({
  runtimeInstalled: true,
  allowDaclFallback: true,
  allowDaclFallbackSaved: true,
  daclLockedByEnvironment: false,
  persistentReadGrants: true,
  persistentReadGrantsSaved: true,
  grantsLockedByEnvironment: false,
  hostPrepMissing: [],
  prepareRepeatsAfterRestart: true,
  ...overrides,
});

const job = (overrides: Partial<HostPrepJob> = {}): HostPrepJob => ({
  id: "j1",
  state: "running",
  startedAt: 1,
  finishedAt: null,
  exitCode: null,
  outputTail: [],
  steps: [],
  ...overrides,
});

test("tool rows name the OS backend when isolated and the reason when not", () => {
  const tool = {
    backend: "macos-seatbelt",
    available: true,
    reason: "",
    limitations: [],
    protectionState: null,
    remediation: "",
  };
  assert.deepEqual(toolRowView(tool), {
    isolated: true,
    backendLabel: "Seatbelt",
    reason: "",
    remediation: "",
    runsInCmd: false,
  });
  assert.equal(
    toolRowView({ ...tool, backend: "bubblewrap" }).backendLabel,
    "bubblewrap",
  );
  assert.equal(
    toolRowView({ ...tool, backend: "mxc-processcontainer" }, "cmd_isolated")
      .runsInCmd,
    true,
  );
  assert.equal(
    toolRowView({ ...tool, backend: "mxc-processcontainer" }, "bash").runsInCmd,
    false,
  );
  const fallback = toolRowView({
    ...tool,
    backend: "none",
    available: false,
    reason: "bwrap: denied",
  });
  assert.equal(fallback.isolated, false);
  assert.equal(fallback.reason, "bwrap: denied");
  // The install command shows only while the tool is not isolated.
  assert.equal(
    toolRowView({ ...tool, available: false, remediation: "run this" })
      .remediation,
    "run this",
  );
  assert.equal(
    toolRowView({ ...tool, remediation: "run this" }).remediation,
    "",
  );
});

test("host preparation reads a lone null-device step as a post-restart repeat", () => {
  assert.equal(hostPrepStatus(windows({ hostPrepMissing: [] })), "prepared");
  assert.equal(
    hostPrepStatus(windows({ hostPrepMissing: ["prepare-null-device"] })),
    "needsPreparingAgain",
  );
  assert.equal(
    hostPrepStatus(
      windows({
        hostPrepMissing: ["prepare-system-drive", "prepare-null-device"],
      }),
    ),
    "needsPreparing",
  );
  assert.equal(
    hostPrepStatus(windows({ hostPrepMissing: ["prepare-system-drive"] })),
    "needsPreparing",
  );
  assert.equal(hostPrepStatus(windows({ hostPrepMissing: null })), "unknown");
  assert.equal(
    hostPrepStatus(
      windows({ allowDaclFallback: false, hostPrepMissing: ["x"] }),
    ),
    "off",
  );
  assert.equal(
    hostPrepStatus(windows({ runtimeInstalled: false })),
    "runtimeMissing",
  );
});

test("the prepare button shows only when something is missing or unknown", () => {
  assert.equal(windowsView(windows(), null, false).showPrepareButton, false);
  assert.equal(
    windowsView(windows({ hostPrepMissing: null }), null, false)
      .showPrepareButton,
    true,
  );
  assert.equal(
    windowsView(
      windows({ hostPrepMissing: ["prepare-null-device"] }),
      null,
      false,
    ).showPrepareButton,
    true,
  );
  assert.equal(
    windowsView(windows({ allowDaclFallback: false }), null, false)
      .showPrepareButton,
    false,
  );
  // A running job keeps its button, disabled, even once the status says prepared.
  const running = windowsView(windows(), job(), false);
  assert.equal(running.showPrepareButton, true);
  assert.equal(running.prepareDisabled, true);
  assert.equal(running.optInDisabled, true);
});

test("environment locks disable their switch and nothing else", () => {
  const locked = windowsView(
    windows({ daclLockedByEnvironment: true }),
    null,
    false,
  );
  assert.equal(locked.optInLocked, true);
  assert.equal(locked.optInDisabled, true);
  assert.equal(locked.grantsDisabled, false);
  const grants = windowsView(
    windows({ grantsLockedByEnvironment: true }),
    null,
    false,
  );
  assert.equal(grants.grantsLocked, true);
  assert.equal(grants.grantsDisabled, true);
  assert.equal(grants.optInDisabled, false);
});

test("the grants switch appears only with the opt-in on and a runtime installed", () => {
  assert.equal(windowsView(windows(), null, false).showGrantsRow, true);
  assert.equal(
    windowsView(windows({ allowDaclFallback: false }), null, false)
      .showGrantsRow,
    false,
  );
  const missing = windowsView(
    windows({ runtimeInstalled: false }),
    null,
    false,
  );
  assert.equal(missing.showGrantsRow, false);
  assert.equal(missing.runtimeMissing, true);
  assert.equal(missing.optInDisabled, true);
});

test("a save in flight disables both switches", () => {
  const saving = windowsView(windows(), null, true);
  assert.equal(saving.optInDisabled, true);
  assert.equal(saving.grantsDisabled, true);
  assert.equal(saving.prepareDisabled, true);
});

test("job results, polling and output lines", () => {
  assert.equal(jobResult(null), null);
  assert.equal(jobResult(job()), null);
  assert.equal(jobResult(job({ state: "idle" })), null);
  assert.equal(jobResult(job({ state: "declined" })), "declined");
  assert.equal(shouldPollJob(job()), true);
  assert.equal(shouldPollJob(job({ state: "succeeded" })), false);
  assert.equal(shouldPollJob(null), false);
  const lines = ["a", "", "b", "c", "d", "e", "f", "g"];
  assert.deepEqual(
    jobOutputLines(job({ state: "failed", outputTail: lines })),
    ["b", "c", "d", "e", "f", "g"],
  );
  assert.deepEqual(
    jobOutputLines(job({ state: "succeeded", outputTail: lines })),
    [],
  );
});
