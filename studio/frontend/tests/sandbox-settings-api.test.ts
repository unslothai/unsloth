// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

type Api = {
  loadSandboxStatus: (
    refresh: boolean,
    fallback: string,
  ) => Promise<Record<string, unknown>>;
  updateSandboxSettings: (
    update: Record<string, boolean>,
    fallback: string,
  ) => Promise<Record<string, unknown>>;
  startHostPreparation: (fallback: string) => Promise<Record<string, unknown>>;
  loadHostPreparation: (fallback: string) => Promise<Record<string, unknown>>;
};

type Call = { url: string; init?: RequestInit };

function loadApi(respond: (call: Call) => Response): {
  api: Api;
  calls: Call[];
} {
  const calls: Call[] = [];
  const api = loadWithStubs<Api>(
    new URL(
      "../src/features/settings/api/sandbox-isolation.ts",
      import.meta.url,
    ),
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
    { relativePassthrough: true },
  );
  return { api, calls };
}

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });

const WINDOWS_STATUS = {
  platform: "win32",
  python: {
    backend: "mxc-processcontainer",
    available: true,
    reason: "passed",
    limitations: ["mxc_preview_not_a_security_boundary"],
    protection_state: "preview",
  },
  terminal: {
    backend: "mxc-processcontainer",
    available: false,
    reason: "bash failed",
    limitations: [],
  },
  terminal_shell: "cmd_isolated",
  windows: {
    runtime_installed: true,
    allow_dacl_fallback: true,
    allow_dacl_fallback_saved: false,
    dacl_locked_by_environment: true,
    persistent_read_grants: false,
    persistent_read_grants_saved: true,
    grants_locked_by_environment: false,
    host_prep_missing: ["prepare-null-device"],
    prepare_repeats_after_restart: true,
  },
  checked_at: 12,
};

test("the status maps to camelCase and keeps the saved and effective values apart", async () => {
  const { api, calls } = loadApi(() => json(WINDOWS_STATUS));
  const status = await api.loadSandboxStatus(true, "fallback");
  assert.equal(calls[0].url, "/api/settings/sandbox?refresh=1");
  assert.deepEqual(status, {
    platform: "win32",
    python: {
      backend: "mxc-processcontainer",
      available: true,
      reason: "passed",
      limitations: ["mxc_preview_not_a_security_boundary"],
      protectionState: "preview",
    },
    terminal: {
      backend: "mxc-processcontainer",
      available: false,
      reason: "bash failed",
      limitations: [],
      protectionState: null,
    },
    terminalShell: "cmd_isolated",
    windows: {
      runtimeInstalled: true,
      allowDaclFallback: true,
      allowDaclFallbackSaved: false,
      daclLockedByEnvironment: true,
      persistentReadGrants: false,
      persistentReadGrantsSaved: true,
      grantsLockedByEnvironment: false,
      hostPrepMissing: ["prepare-null-device"],
      prepareRepeatsAfterRestart: true,
    },
    checkedAt: 12,
  });
});

test("a non-Windows status has no Windows block and a plain load skips refresh", async () => {
  const { api, calls } = loadApi(() =>
    json({
      platform: "linux",
      python: {
        backend: "bubblewrap",
        available: true,
        reason: "",
        limitations: [],
      },
      terminal: {
        backend: "bubblewrap",
        available: true,
        reason: "",
        limitations: [],
      },
      terminal_shell: null,
      windows: null,
      checked_at: 1,
    }),
  );
  const status = await api.loadSandboxStatus(false, "fallback");
  assert.equal(calls[0].url, "/api/settings/sandbox");
  assert.equal(status.windows, null);
  assert.equal(status.terminalShell, null);
});

test("an unknown host preparation stays null rather than reading as prepared", async () => {
  const { api } = loadApi(() =>
    json({
      ...WINDOWS_STATUS,
      windows: { ...WINDOWS_STATUS.windows, host_prep_missing: null },
    }),
  );
  const status = await api.loadSandboxStatus(false, "fallback");
  assert.equal(
    (status.windows as { hostPrepMissing: unknown }).hostPrepMissing,
    null,
  );
});

test("a save sends only the fields it changes, in snake_case, and reports restored grants", async () => {
  const { api, calls } = loadApi(() =>
    json({ ...WINDOWS_STATUS, restored: 2 }),
  );
  const status = await api.updateSandboxSettings(
    { allowDaclFallback: false },
    "fallback",
  );
  assert.equal(calls[0].init?.method, "PUT");
  assert.deepEqual(JSON.parse(String(calls[0].init?.body)), {
    allow_dacl_fallback: false,
  });
  assert.equal(status.restored, 2);
});

test("an older backend without the routes is told apart from a failure", async () => {
  const { api } = loadApi(() => new Response(null, { status: 404 }));
  await assert.rejects(api.loadSandboxStatus(false, "fallback"), {
    name: "SettingsRouteAbsentError",
  });
  await assert.rejects(api.startHostPreparation("fallback"), {
    name: "SettingsRouteAbsentError",
  });
});

test("a refused request surfaces the server's reason, else the fallback", async () => {
  const { api } = loadApi((call) =>
    call.init?.method === "POST"
      ? json(
          { detail: "Open Unsloth on the computer running it to prepare it." },
          403,
        )
      : new Response("boom", { status: 500 }),
  );
  await assert.rejects(
    api.startHostPreparation("fallback"),
    /on the computer running it/,
  );
  await assert.rejects(
    api.loadHostPreparation("Failed to prepare"),
    /Failed to prepare/,
  );
});

test("a job maps its fields and an idle answer carries no id", async () => {
  const { api } = loadApi((call) =>
    call.init?.method === "POST"
      ? json({
          id: "j1",
          state: "running",
          started_at: 5,
          finished_at: null,
          exit_code: null,
          output_tail: ["preparing"],
          steps: ["prepare-null-device"],
        })
      : json({ state: "idle" }),
  );
  assert.deepEqual(await api.startHostPreparation("fallback"), {
    id: "j1",
    state: "running",
    startedAt: 5,
    finishedAt: null,
    exitCode: null,
    outputTail: ["preparing"],
    steps: ["prepare-null-device"],
  });
  const idle = await api.loadHostPreparation("fallback");
  assert.equal(idle.state, "idle");
  assert.equal(idle.id, null);
});
