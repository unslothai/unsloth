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
  isOlderJob,
  jobOutputLines,
  jobResult,
  shouldPollJob,
  toolRowView,
  toolRowsQuiet,
  windowsView,
} from "../src/features/settings/tabs/sandbox-tab-state.ts";

const windows = (
  overrides: Partial<WindowsSandboxStatus> = {},
): WindowsSandboxStatus => ({
  runtimeInstalled: true,
  runtimeUnsupported: null,
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

test("a late read of an earlier job never replaces the one this tab started", () => {
  const started = job({ id: "j2", startedAt: 20 });
  const stale = job({ id: "j1", state: "failed", startedAt: 10 });
  assert.equal(isOlderJob(stale, started), true);
  assert.equal(isOlderJob(stale, null), false);
  assert.equal(
    isOlderJob(job({ id: "j2", state: "succeeded", startedAt: 20 }), started),
    false,
  );
  assert.equal(isOlderJob(job({ id: "j3", startedAt: 30 }), started), false);
});

test("the setup job's first read is held to the same rule", () => {
  const started = { id: "s2", startedAt: 20 };
  assert.equal(isOlderJob({ id: "s1", startedAt: 10 }, started), true);
  assert.equal(isOlderJob({ id: "s2", startedAt: 20 }, started), false);
});

test("a missing runtime offers the install unless this Windows cannot run MXC", () => {
  const missing = windowsView(windows({ runtimeInstalled: false }), null, false);
  assert.equal(missing.runtimeMissing, true);
  assert.equal(missing.showInstallRuntime, true);
  assert.equal(missing.unsupported, null);
  for (const reason of ["arch", "build"] as const) {
    const view = windowsView(
      windows({ runtimeInstalled: false, runtimeUnsupported: reason }),
      null,
      false,
    );
    assert.equal(view.unsupported, reason);
    assert.equal(view.showInstallRuntime, false);
  }
  assert.equal(
    windowsView(windows({ runtimeUnsupported: "build" }), null, false)
      .unsupported,
    "build",
  );
});

test("the tool rows keep only their badge while a section below has the answer", () => {
  assert.equal(toolRowsQuiet(true, null), true);
  assert.equal(toolRowsQuiet(false, null), false);
  const view = (overrides: Partial<WindowsSandboxStatus>) =>
    windowsView(windows(overrides), null, false);
  assert.equal(toolRowsQuiet(false, view({ runtimeInstalled: false })), true);
  assert.equal(toolRowsQuiet(false, view({ allowDaclFallback: false })), true);
  assert.equal(
    toolRowsQuiet(false, view({ hostPrepMissing: ["prepare-null-device"] })),
    true,
  );
  assert.equal(toolRowsQuiet(false, view({ runtimeUnsupported: "build" })), true);
  assert.equal(toolRowsQuiet(false, view({})), false);
});
