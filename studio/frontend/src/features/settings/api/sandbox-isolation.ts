// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

import { SettingsRouteAbsentError } from "./settings-route-absent";

const ROUTE = "/api/settings/sandbox";
const PREPARE_ROUTE = "/api/settings/sandbox/prepare";

export type SandboxToolStatus = {
  backend: string;
  available: boolean;
  reason: string;
  limitations: string[];
  protectionState: string | null;
  remediation: string;
};

export type TerminalShell = "bash" | "cmd_isolated" | "cmd_fallback";

// "arch": not x64 Windows; "build": older than Windows 11 build 26100. MXC cannot run on either.
export type RuntimeUnsupported = "arch" | "build";

export type WindowsSandboxStatus = {
  runtimeInstalled: boolean;
  runtimeUnsupported: RuntimeUnsupported | null;
  allowDaclFallback: boolean;
  allowDaclFallbackSaved: boolean;
  daclLockedByEnvironment: boolean;
  persistentReadGrants: boolean;
  persistentReadGrantsSaved: boolean;
  grantsLockedByEnvironment: boolean;
  // null: MXC could not tell which host preparation steps are missing.
  hostPrepMissing: string[] | null;
  prepareRepeatsAfterRestart: boolean;
  // true: MXC runs in Windows' built-in container; false: this Windows has none; null: unknown.
  builtinContainer: boolean | null;
};

export type SandboxSetupPlan = {
  action: "linux-install" | "windows-setup" | null;
  elevation: string | null;
  manualCommand: string;
  reason: string;
  canRun: boolean;
};

export type SandboxStatus = {
  platform: string;
  python: SandboxToolStatus;
  terminal: SandboxToolStatus;
  terminalShell: TerminalShell | null;
  windows: WindowsSandboxStatus | null;
  setup: SandboxSetupPlan | null;
  checkedAt: number;
  restored?: number;
};

export type SandboxSettingsUpdate = {
  allowDaclFallback?: boolean;
  persistentReadGrants?: boolean;
};

export type HostPrepState =
  | "idle"
  | "running"
  | "succeeded"
  | "declined"
  | "failed";

export type HostPrepJob = {
  id: string | null;
  state: HostPrepState;
  startedAt: number | null;
  finishedAt: number | null;
  exitCode: number | null;
  outputTail: string[];
  steps: string[];
};

type ApiToolStatus = {
  backend?: string;
  available?: boolean;
  reason?: string;
  limitations?: string[];
  // biome-ignore lint/style/useNamingConvention: API schema
  protection_state?: string | null;
  remediation?: string | null;
};

type ApiSetupPlan = {
  action?: string | null;
  elevation?: string | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  manual_command?: string | null;
  reason?: string | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  can_run?: boolean;
};

type ApiWindowsStatus = {
  // biome-ignore lint/style/useNamingConvention: API schema
  runtime_installed?: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  runtime_unsupported?: string | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  allow_dacl_fallback?: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  allow_dacl_fallback_saved?: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  dacl_locked_by_environment?: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  persistent_read_grants?: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  persistent_read_grants_saved?: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  grants_locked_by_environment?: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  host_prep_missing?: string[] | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  prepare_repeats_after_restart?: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  builtin_container?: boolean | null;
};

type ApiSandboxStatus = {
  platform?: string;
  python?: ApiToolStatus;
  terminal?: ApiToolStatus;
  // biome-ignore lint/style/useNamingConvention: API schema
  terminal_shell?: TerminalShell | null;
  windows?: ApiWindowsStatus | null;
  setup?: ApiSetupPlan | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  checked_at?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  grants_restored?: number | null;
};

type ApiHostPrepJob = {
  id?: string | null;
  state?: HostPrepState;
  // biome-ignore lint/style/useNamingConvention: API schema
  started_at?: number | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  finished_at?: number | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  exit_code?: number | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  output_tail?: string[];
  steps?: string[];
};

function toolFromApi(tool: ApiToolStatus | undefined): SandboxToolStatus {
  return {
    backend: tool?.backend ?? "none",
    available: tool?.available ?? false,
    reason: tool?.reason ?? "",
    limitations: tool?.limitations ?? [],
    protectionState: tool?.protection_state ?? null,
    remediation: tool?.remediation ?? "",
  };
}

function setupFromApi(setup: ApiSetupPlan | null | undefined): SandboxSetupPlan | null {
  if (!setup) return null;
  const action =
    setup.action === "linux-install" || setup.action === "windows-setup"
      ? setup.action
      : null;
  return {
    action,
    elevation: setup.elevation ?? null,
    manualCommand: setup.manual_command ?? "",
    reason: setup.reason ?? "",
    canRun: action !== null && (setup.can_run ?? true),
  };
}

function windowsFromApi(
  windows: ApiWindowsStatus | null | undefined,
): WindowsSandboxStatus | null {
  if (!windows) return null;
  return {
    runtimeInstalled: windows.runtime_installed ?? false,
    runtimeUnsupported:
      windows.runtime_unsupported === "arch" ||
      windows.runtime_unsupported === "build"
        ? windows.runtime_unsupported
        : null,
    allowDaclFallback: windows.allow_dacl_fallback ?? false,
    allowDaclFallbackSaved: windows.allow_dacl_fallback_saved ?? false,
    daclLockedByEnvironment: windows.dacl_locked_by_environment ?? false,
    persistentReadGrants: windows.persistent_read_grants ?? true,
    persistentReadGrantsSaved: windows.persistent_read_grants_saved ?? true,
    grantsLockedByEnvironment: windows.grants_locked_by_environment ?? false,
    hostPrepMissing:
      windows.host_prep_missing === undefined
        ? null
        : windows.host_prep_missing,
    prepareRepeatsAfterRestart: windows.prepare_repeats_after_restart ?? true,
    builtinContainer: windows.builtin_container ?? null,
  };
}

export function statusFromApi(status: ApiSandboxStatus): SandboxStatus {
  const out: SandboxStatus = {
    platform: status.platform ?? "",
    python: toolFromApi(status.python),
    terminal: toolFromApi(status.terminal),
    terminalShell: status.terminal_shell ?? null,
    windows: windowsFromApi(status.windows),
    setup: setupFromApi(status.setup),
    checkedAt: status.checked_at ?? 0,
  };
  if (typeof status.grants_restored === "number") out.restored = status.grants_restored;
  return out;
}

export function jobFromApi(job: ApiHostPrepJob): HostPrepJob {
  return {
    id: job.id ?? null,
    state: job.state ?? "idle",
    startedAt: job.started_at ?? null,
    finishedAt: job.finished_at ?? null,
    exitCode: job.exit_code ?? null,
    outputTail: job.output_tail ?? [],
    steps: job.steps ?? [],
  };
}

// Older backends lack these routes: reported apart from a failure so the tab can say so.
async function checked(
  res: Response,
  route: string,
  fallbackMessage: string,
): Promise<Response> {
  if (res.status === 404) {
    throw new SettingsRouteAbsentError(route);
  }
  if (!res.ok) {
    throw new Error(await readFastApiError(res, fallbackMessage));
  }
  return res;
}

export async function loadSandboxStatus(
  refresh: boolean,
  fallbackMessage: string,
): Promise<SandboxStatus> {
  const res = await authFetch(refresh ? `${ROUTE}?refresh=1` : ROUTE);
  await checked(res, ROUTE, fallbackMessage);
  return statusFromApi(await res.json());
}

export async function updateSandboxSettings(
  update: SandboxSettingsUpdate,
  fallbackMessage: string,
): Promise<SandboxStatus> {
  const body: Record<string, boolean> = {};
  if (update.allowDaclFallback !== undefined) {
    body.allow_dacl_fallback = update.allowDaclFallback;
  }
  if (update.persistentReadGrants !== undefined) {
    body.persistent_read_grants = update.persistentReadGrants;
  }
  const res = await authFetch(ROUTE, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  await checked(res, ROUTE, fallbackMessage);
  return statusFromApi(await res.json());
}

export async function startHostPreparation(
  fallbackMessage: string,
): Promise<HostPrepJob> {
  const res = await authFetch(PREPARE_ROUTE, { method: "POST" });
  await checked(res, PREPARE_ROUTE, fallbackMessage);
  return jobFromApi(await res.json());
}

export async function loadHostPreparation(
  fallbackMessage: string,
): Promise<HostPrepJob> {
  const res = await authFetch(PREPARE_ROUTE);
  await checked(res, PREPARE_ROUTE, fallbackMessage);
  return jobFromApi(await res.json());
}
