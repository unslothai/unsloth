// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type {
  SandboxCapability,
  SandboxSetupJob,
} from "../src/features/chat/api/sandbox-capability.ts";
import type { SandboxStatus } from "../src/features/settings/api/sandbox-isolation.ts";
import {
  canInstallWindowsRuntime,
  setupRowView,
} from "../src/features/settings/tabs/sandbox-tab-state.ts";
import * as tabState from "../src/features/settings/tabs/sandbox-tab-state.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

type Call = { url: string; init?: RequestInit };

type CapabilityApi = {
  loadSandboxCapability: (options?: {
    force?: boolean;
  }) => Promise<SandboxCapability | null>;
  forgetSandboxCapability: () => void;
  sandboxReady: (capability: SandboxCapability) => boolean;
  startSandboxSetup: (
    operation: string,
    options: { consentDaclFallback?: boolean },
    fallback: string,
  ) => Promise<SandboxSetupJob>;
  loadSandboxSetup: (fallback: string) => Promise<SandboxSetupJob>;
  loadSettledSandboxCapability: (options?: {
    attempts?: number;
    delayMs?: number;
  }) => Promise<SandboxCapability | null>;
};

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });

function loadCapabilityApi(respond: (call: Call) => Response) {
  const calls: Call[] = [];
  const api = loadWithStubs<CapabilityApi>(
    new URL("../src/features/chat/api/sandbox-capability.ts", import.meta.url),
    {
      "@/features/auth": {
        authFetch: async (url: string, init?: RequestInit) => {
          const call = { url, init };
          calls.push(call);
          return respond(call);
        },
      },
      "@/lib/format-fastapi-error": {
        readFastApiError: async (response: Response, fallback: string) => {
          const body = (await response.json().catch(() => null)) as {
            detail?: string;
          } | null;
          return body?.detail ?? fallback;
        },
      },
    },
  );
  return { api, calls };
}

const LINUX_UNAVAILABLE = {
  python_os_isolated: false,
  terminal_os_isolated: false,
  backend: "bubblewrap",
  platform: "linux",
  reason: "bwrap: setting up uid map: Permission denied",
  setup_action: "linux-install",
  manual_command: "apt-get install -y apparmor-profiles",
  can_run_setup: true,
};

test("the capability maps to camelCase and is cached between picker opens", async () => {
  const { api, calls } = loadCapabilityApi(() => json(LINUX_UNAVAILABLE));
  const first = await api.loadSandboxCapability();
  assert.deepEqual(first, {
    pythonOsIsolated: false,
    terminalOsIsolated: false,
    backend: "bubblewrap",
    platform: "linux",
    reason: "bwrap: setting up uid map: Permission denied",
    setupAction: "linux-install",
    manualCommand: "apt-get install -y apparmor-profiles",
    canRunSetup: true,
    needsConsent: false,
  });
  await api.loadSandboxCapability();
  assert.equal(calls.length, 1);
  await api.loadSandboxCapability({ force: true });
  assert.equal(calls.length, 2);
  assert.equal(calls[1].url, "/api/sandbox/capability?refresh=1");
  assert.equal(api.sandboxReady(first as SandboxCapability), false);
});

test("an older server (404) or a failed check leaves the capability unknown", async () => {
  for (const response of [json({ detail: "Not Found" }, 404), json({}, 500)]) {
    const { api } = loadCapabilityApi(() => response);
    assert.equal(await api.loadSandboxCapability(), null);
  }
  const { api } = loadCapabilityApi(() => {
    throw new Error("offline");
  });
  assert.equal(await api.loadSandboxCapability(), null);
});

test("a setup the server did not name cannot be run, whatever the flag says", async () => {
  const { api } = loadCapabilityApi(() =>
    json({
      ...LINUX_UNAVAILABLE,
      setup_action: "curl-bash",
      can_run_setup: true,
    }),
  );
  const capability = await api.loadSandboxCapability();
  assert.equal(capability?.setupAction, null);
  assert.equal(capability?.canRunSetup, false);
});

test("starting a setup sends only the operation and the consent", async () => {
  const { api, calls } = loadCapabilityApi(() =>
    json({
      id: "s1",
      operation: "windows-setup",
      state: "running",
      output_tail: [],
    }),
  );
  const job = await api.startSandboxSetup(
    "windows-setup",
    { consentDaclFallback: true },
    "fallback",
  );
  assert.equal(calls[0].url, "/api/settings/sandbox/setup");
  assert.equal(calls[0].init?.method, "POST");
  assert.deepEqual(JSON.parse(String(calls[0].init?.body)), {
    operation: "windows-setup",
    consent_dacl_fallback: true,
  });
  assert.equal(job.state, "running");
  assert.equal(job.operation, "windows-setup");
});

test("a refused setup surfaces the server's reason", async () => {
  const { api } = loadCapabilityApi(() =>
    json(
      { detail: "Set up the sandbox from the computer running Unsloth." },
      403,
    ),
  );
  await assert.rejects(
    api.startSandboxSetup("linux-install", {}, "fallback"),
    /computer running Unsloth/,
  );
});

test("Windows consent is asked only while the opt-in is off, and only of who can run it", async () => {
  const windows = {
    ...LINUX_UNAVAILABLE,
    platform: "win32",
    setup_action: "windows-setup",
    needs_consent: true,
  };
  for (const [body, expected] of [
    [windows, true],
    [{ ...windows, needs_consent: false }, false],
    [{ ...windows, can_run_setup: false }, false],
    [{ ...LINUX_UNAVAILABLE, needs_consent: true }, false],
  ] as const) {
    const { api } = loadCapabilityApi(() => json(body));
    assert.equal((await api.loadSandboxCapability())?.needsConsent, expected);
  }
});

test("the runtime-only operation is sent as is and the job note is kept", async () => {
  const { api, calls } = loadCapabilityApi(() =>
    json({
      id: "s2",
      operation: "windows-runtime",
      state: "failed",
      note: "a password is required",
      manual_command: "python install_mxc_prebuilt.py",
    }),
  );
  const job = await api.startSandboxSetup("windows-runtime", {}, "fallback");
  assert.deepEqual(JSON.parse(String(calls[0].init?.body)), {
    operation: "windows-runtime",
    consent_dacl_fallback: false,
  });
  assert.equal(job.note, "a password is required");
  assert.equal(job.manualCommand, "python install_mxc_prebuilt.py");
});

test("a settled read waits out a check that has not answered yet, but not a real no", async () => {
  const pending = { ...LINUX_UNAVAILABLE, backend: "unknown" };
  const ready = {
    ...LINUX_UNAVAILABLE,
    python_os_isolated: true,
    terminal_os_isolated: true,
  };
  let n = 0;
  const { api, calls } = loadCapabilityApi(() =>
    json(n++ < 2 ? pending : ready),
  );
  const settled = await api.loadSettledSandboxCapability({ delayMs: 1 });
  assert.equal(calls.length, 3);
  assert.ok(calls.every((call) => call.url.endsWith("?refresh=1")));
  assert.equal(settled && api.sandboxReady(settled), true);

  const denied = loadCapabilityApi(() => json(LINUX_UNAVAILABLE));
  const no = await denied.api.loadSettledSandboxCapability({ delayMs: 1 });
  assert.equal(denied.calls.length, 1);
  assert.equal(no?.backend, "bubblewrap");

  const stuck = loadCapabilityApi(() => json(pending));
  await stuck.api.loadSettledSandboxCapability({ attempts: 2, delayMs: 1 });
  assert.equal(stuck.calls.length, 2);
});

// ---- picking "Full access in sandbox" ----

type PickModule = typeof import("../src/features/chat/sandbox-pick.ts");

const pickModule = loadWithStubs<PickModule>(
  new URL("../src/features/chat/sandbox-pick.ts", import.meta.url),
  {
    "./api/sandbox-capability": {
      loadSandboxCapability: async () => null,
      sandboxReady: (c: SandboxCapability) =>
        c.pythonOsIsolated && c.terminalOsIsolated,
    },
  },
);

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}

test("a slow capability read does not undo a mode picked while it was in flight", async () => {
  let mode: string = "auto";
  const applied: string[] = [];
  let dialogs = 0;
  const read = deferred<SandboxCapability | null>();
  const pending = pickModule.pickSandboxedMode(
    (next) => {
      applied.push(next);
      mode = next;
    },
    () => dialogs++,
    () => mode as never,
    () => read.promise,
  );
  mode = "ask";
  read.resolve(null);
  await pending;
  assert.deepEqual(applied, []);
  assert.equal(dialogs, 0);
});

test("only the latest pick acts; a missing sandbox opens the setup instead of applying", async () => {
  let mode: string = "auto";
  const applied: string[] = [];
  let dialogs = 0;
  const set = (next: string) => {
    applied.push(next);
    mode = next;
  };
  const first = deferred<SandboxCapability | null>();
  const older = pickModule.pickSandboxedMode(
    set as never,
    () => dialogs++,
    () => mode as never,
    () => first.promise,
  );
  const newer = pickModule.pickSandboxedMode(
    set as never,
    () => dialogs++,
    () => mode as never,
    async () => capability({ pythonOsIsolated: false }),
  );
  await newer;
  first.resolve(null);
  await older;
  assert.equal(dialogs, 1);
  assert.deepEqual(applied, []);
  await pickModule.pickSandboxedMode(
    set as never,
    () => dialogs++,
    () => mode as never,
    async () =>
      capability({ pythonOsIsolated: true, terminalOsIsolated: true }),
  );
  assert.deepEqual(applied, ["off"]);
});

// ---- the dialog's pure view ----

type SetupState = typeof import("../src/features/chat/sandbox-setup-state.ts");

const setupState = loadWithStubs<SetupState>(
  new URL("../src/features/chat/sandbox-setup-state.ts", import.meta.url),
  { "@/features/settings": tabState },
);

const capability = (
  overrides: Partial<SandboxCapability> = {},
): SandboxCapability => ({
  pythonOsIsolated: false,
  terminalOsIsolated: false,
  backend: "bubblewrap",
  platform: "linux",
  reason: "denied",
  setupAction: "linux-install",
  manualCommand: "apt-get install -y bubblewrap",
  canRunSetup: true,
  needsConsent: false,
  ...overrides,
});

const setupJob = (
  overrides: Partial<SandboxSetupJob> = {},
): SandboxSetupJob => ({
  id: "s1",
  operation: "linux-install",
  state: "running",
  startedAt: 1,
  finishedAt: null,
  exitCode: null,
  outputTail: [],
  steps: [],
  manualCommand: "",
  note: "",
  ...overrides,
});

test("the dialog shows a check in progress before the capability arrives", () => {
  const view = setupState.sandboxSetupView({
    capability: null,
    job: null,
    consent: false,
  });
  assert.equal(view.checking, true);
  assert.equal(view.install, null);
  assert.equal(view.command, "");
});

test("the owner on Linux gets Install sandbox and the command", () => {
  const view = setupState.sandboxSetupView({
    capability: capability(),
    job: null,
    consent: false,
  });
  assert.equal(view.install, "linux");
  assert.equal(view.installDisabled, false);
  assert.equal(view.showConsent, false);
  assert.equal(view.command, "apt-get install -y bubblewrap");
  assert.equal(view.showOwnerOnly, false);
});

test("Windows needs the MXC consent before the setup can start", () => {
  const windows = capability({
    platform: "win32",
    backend: "mxc-processcontainer",
    setupAction: "windows-setup",
    manualCommand: "",
    needsConsent: true,
  });
  const without = setupState.sandboxSetupView({
    capability: windows,
    job: null,
    consent: false,
  });
  assert.equal(without.install, "windows");
  assert.equal(without.showConsent, true);
  assert.equal(without.installDisabled, true);
  assert.equal(without.command, "");
  const withConsent = setupState.sandboxSetupView({
    capability: windows,
    job: null,
    consent: true,
  });
  assert.equal(withConsent.installDisabled, false);
});

test("someone who cannot run the setup gets the command and who can", () => {
  const view = setupState.sandboxSetupView({
    capability: capability({ canRunSetup: false }),
    job: null,
    consent: false,
  });
  assert.equal(view.install, null);
  assert.equal(view.showOwnerOnly, true);
  // Nothing to install and nothing to run (e.g. macOS): no owner note either.
  const macos = setupState.sandboxSetupView({
    capability: capability({
      platform: "darwin",
      setupAction: null,
      canRunSetup: false,
      manualCommand: "",
    }),
    job: null,
    consent: false,
  });
  assert.equal(macos.showOwnerOnly, false);
  assert.equal(macos.install, null);
});

test("a running job disables the button; a failure shows its output and command", () => {
  const running = setupState.sandboxSetupView({
    capability: capability(),
    job: setupJob(),
    consent: false,
  });
  assert.equal(running.running, true);
  assert.equal(running.installDisabled, true);
  const failed = setupState.sandboxSetupView({
    capability: capability(),
    job: setupJob({
      state: "failed",
      exitCode: 100,
      outputTail: ["", "E: Unable to locate package bubblewrap"],
      manualCommand: "apt-get update && apt-get install -y bubblewrap",
    }),
    consent: false,
  });
  assert.equal(failed.result, "failed");
  assert.deepEqual(failed.outputLines, [
    "E: Unable to locate package bubblewrap",
  ]);
  assert.equal(
    failed.command,
    "apt-get update && apt-get install -y bubblewrap",
  );
  const declined = setupState.sandboxSetupView({
    capability: capability(),
    job: setupJob({ state: "declined", exitCode: 126 }),
    consent: false,
  });
  assert.equal(declined.result, "declined");
  assert.equal(declined.command, "apt-get install -y bubblewrap");
});

test("a failed check shows Retry instead of an endless spinner", () => {
  const view = setupState.sandboxSetupView({
    capability: null,
    job: null,
    consent: false,
    loadFailed: true,
  });
  assert.equal(view.checking, false);
  assert.equal(view.loadFailed, true);
  assert.equal(view.install, null);
});

test("Windows skips the consent once the opt-in is on (after a reboot, only preparation)", () => {
  const view = setupState.sandboxSetupView({
    capability: capability({
      platform: "win32",
      setupAction: "windows-setup",
      needsConsent: false,
    }),
    job: null,
    consent: false,
  });
  assert.equal(view.install, "windows");
  assert.equal(view.showConsent, false);
  assert.equal(view.installDisabled, false);
});

test("the server's note shows for a declined or failed setup only", () => {
  const note = "Not authorized, or no authentication agent";
  for (const [state, shown] of [
    ["failed", note],
    ["declined", note],
    ["running", ""],
    ["succeeded", ""],
  ] as const) {
    const view = setupState.sandboxSetupView({
      capability: capability(),
      job: setupJob({ state, note }),
      consent: false,
    });
    assert.equal(view.note, shown, state);
  }
});

// ---- the Settings tab setup row ----

const status = (overrides: Partial<SandboxStatus> = {}): SandboxStatus => ({
  platform: "linux",
  python: {
    backend: "bubblewrap",
    available: false,
    reason: "missing",
    limitations: [],
    protectionState: null,
    remediation: "",
  },
  terminal: {
    backend: "bubblewrap",
    available: false,
    reason: "missing",
    limitations: [],
    protectionState: null,
    remediation: "",
  },
  terminalShell: "bash",
  windows: null,
  setup: {
    action: "linux-install",
    elevation: "sudo",
    manualCommand: "apt-get install -y bubblewrap",
    reason: "bubblewrap is not installed",
    canRun: true,
  },
  checkedAt: 1,
  ...overrides,
});

test("Linux shows Install sandbox with the command; a remote owner only the command", () => {
  const row = setupRowView(status(), null);
  assert.equal(row.show, true);
  assert.equal(row.showInstall, true);
  assert.equal(row.command, "apt-get install -y bubblewrap");
  const remote = setupRowView(
    status({ setup: { ...status().setup!, canRun: false } }),
    null,
  );
  assert.equal(remote.showInstall, false);
  assert.equal(remote.command, "apt-get install -y bubblewrap");
});

test("a working Linux sandbox, an older server and Windows hide the row", () => {
  assert.equal(
    setupRowView(
      status({
        setup: {
          action: null,
          elevation: null,
          manualCommand: "",
          reason: "",
          canRun: false,
        },
      }),
      null,
    ).show,
    false,
  );
  assert.equal(setupRowView(status({ setup: null }), null).show, false);
  assert.equal(setupRowView(status({ platform: "win32" }), null).show, false);
});

test("macOS explains Seatbelt instead of offering an install, and only when it fails", () => {
  const failing = setupRowView(
    status({ platform: "darwin", setup: null }),
    null,
  );
  assert.equal(failing.show, true);
  assert.equal(failing.builtIn, true);
  assert.equal(failing.showInstall, false);
  assert.equal(failing.reason, "missing");
  const working = status({ platform: "darwin", setup: null });
  working.python.available = true;
  working.terminal.available = true;
  assert.equal(setupRowView(working, null).show, false);
});

test("a failed install shows the job's command; a running one keeps the button", () => {
  const failed = setupRowView(
    status(),
    { ...setupJob({ state: "failed" }) },
    "apt-get update && apt-get install -y bubblewrap",
  );
  assert.equal(
    failed.command,
    "apt-get update && apt-get install -y bubblewrap",
  );
  const running = setupRowView(
    status({ setup: { ...status().setup!, canRun: false, manualCommand: "" } }),
    setupJob(),
  );
  assert.equal(running.show, true);
  assert.equal(running.showInstall, true);
  assert.equal(running.installDisabled, true);
});

test("the Windows runtime install is offered only when missing and allowed", () => {
  const windowsBlock = {
    runtimeInstalled: false,
    allowDaclFallback: false,
    allowDaclFallbackSaved: false,
    daclLockedByEnvironment: false,
    persistentReadGrants: true,
    persistentReadGrantsSaved: true,
    grantsLockedByEnvironment: false,
    hostPrepMissing: null,
    prepareRepeatsAfterRestart: true,
  };
  const setup = {
    action: "windows-setup" as const,
    elevation: "uac",
    manualCommand: "",
    reason: "",
    canRun: true,
  };
  assert.equal(
    canInstallWindowsRuntime(
      status({ platform: "win32", windows: windowsBlock, setup }),
    ),
    true,
  );
  assert.equal(
    canInstallWindowsRuntime(
      status({
        platform: "win32",
        windows: { ...windowsBlock, runtimeInstalled: true },
        setup,
      }),
    ),
    false,
  );
  assert.equal(
    canInstallWindowsRuntime(
      status({
        platform: "win32",
        windows: windowsBlock,
        setup: { ...setup, canRun: false },
      }),
    ),
    false,
  );
});
