// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

const CAPABILITY_ROUTE = "/api/sandbox/capability";
const SETUP_ROUTE = "/api/settings/sandbox/setup";
const CAPABILITY_TTL_MS = 30_000;

export type SandboxSetupAction = "linux-install" | "windows-setup";
export type SandboxSetupOperation = SandboxSetupAction | "windows-runtime";

export type SandboxCapability = {
  pythonOsIsolated: boolean;
  terminalOsIsolated: boolean;
  backend: string;
  platform: string;
  reason: string;
  setupAction: SandboxSetupAction | null;
  manualCommand: string;
  canRunSetup: boolean;
  // "no_elevation": owner is local but there is no passwordless sudo or desktop prompt.
  setupBlocked: "not_owner" | "not_local" | "no_elevation" | null;
  needsConsent: boolean;
};

export type SandboxSetupState =
  | "idle"
  | "running"
  | "succeeded"
  | "declined"
  | "failed";

export type SandboxSetupJob = {
  id: string | null;
  operation: string | null;
  state: SandboxSetupState;
  startedAt: number | null;
  finishedAt: number | null;
  exitCode: number | null;
  outputTail: string[];
  steps: string[];
  manualCommand: string;
  note: string;
};

type ApiCapability = {
  // biome-ignore lint/style/useNamingConvention: API schema
  python_os_isolated?: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  terminal_os_isolated?: boolean;
  backend?: string;
  platform?: string;
  reason?: string;
  // biome-ignore lint/style/useNamingConvention: API schema
  setup_action?: string | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  manual_command?: string | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  can_run_setup?: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  setup_blocked?: string | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  needs_consent?: boolean;
};

type ApiSetupJob = {
  id?: string | null;
  operation?: string | null;
  state?: SandboxSetupState;
  // biome-ignore lint/style/useNamingConvention: API schema
  started_at?: number | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  finished_at?: number | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  exit_code?: number | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  output_tail?: string[];
  steps?: string[];
  // biome-ignore lint/style/useNamingConvention: API schema
  manual_command?: string | null;
  note?: string | null;
};

function setupActionFromApi(
  value: string | null | undefined,
): SandboxSetupAction | null {
  return value === "linux-install" || value === "windows-setup" ? value : null;
}

export function capabilityFromApi(body: ApiCapability): SandboxCapability {
  const setupAction = setupActionFromApi(body.setup_action);
  const canRunSetup = setupAction !== null && (body.can_run_setup ?? false);
  return {
    pythonOsIsolated: body.python_os_isolated ?? false,
    terminalOsIsolated: body.terminal_os_isolated ?? false,
    backend: body.backend ?? "none",
    platform: body.platform ?? "",
    reason: body.reason ?? "",
    setupAction,
    manualCommand: body.manual_command ?? "",
    canRunSetup,
    setupBlocked:
      body.setup_blocked === "not_owner" ||
      body.setup_blocked === "not_local" ||
      body.setup_blocked === "no_elevation"
        ? body.setup_blocked
        : null,
    needsConsent:
      canRunSetup &&
      setupAction === "windows-setup" &&
      (body.needs_consent ?? false),
  };
}

export function setupJobFromApi(job: ApiSetupJob): SandboxSetupJob {
  return {
    id: job.id ?? null,
    operation: job.operation ?? null,
    state: job.state ?? "idle",
    startedAt: job.started_at ?? null,
    finishedAt: job.finished_at ?? null,
    exitCode: job.exit_code ?? null,
    outputTail: job.output_tail ?? [],
    steps: job.steps ?? [],
    manualCommand: job.manual_command ?? "",
    note: job.note ?? "",
  };
}

export function sandboxReady(capability: SandboxCapability): boolean {
  return capability.pythonOsIsolated && capability.terminalOsIsolated;
}

export function capabilityPending(capability: SandboxCapability): boolean {
  return capability.backend === "unknown" && !sandboxReady(capability);
}

let cached: { at: number; value: SandboxCapability | null } | null = null;
let inFlight: Promise<SandboxCapability | null> | null = null;
// Only the latest request writes the cache: an older one finishing last must not replace it.
let latestRequest = 0;
const changeListeners = new Set<() => void>();

export function onSandboxCapabilityChange(listener: () => void): () => void {
  changeListeners.add(listener);
  return () => {
    changeListeners.delete(listener);
  };
}

function notifyCapabilityChange(): void {
  for (const listener of [...changeListeners]) listener();
}

export function loadSandboxCapability({
  force = false,
}: { force?: boolean } = {}): Promise<SandboxCapability | null> {
  if (!force && cached && Date.now() - cached.at < CAPABILITY_TTL_MS) {
    return Promise.resolve(cached.value);
  }
  if (!force && inFlight) return inFlight;
  const id = ++latestRequest;
  const request = authFetch(
    force ? `${CAPABILITY_ROUTE}?refresh=1` : CAPABILITY_ROUTE,
  )
    .then(async (res) => {
      if (!res.ok) return null;
      return capabilityFromApi((await res.json()) as ApiCapability);
    })
    .catch(() => null)
    .then((value) => {
      if (id === latestRequest) {
        cached = { at: Date.now(), value };
        notifyCapabilityChange();
      }
      return value;
    })
    .finally(() => {
      if (inFlight === request) inFlight = null;
    });
  inFlight = request;
  return request;
}

const PENDING_RETRY_MS = 1500;

/** A forced read that retries a few times while the backend is still "unknown". */
export async function loadSettledSandboxCapability({
  attempts = 3,
  delayMs = PENDING_RETRY_MS,
}: {
  attempts?: number;
  delayMs?: number;
} = {}): Promise<SandboxCapability | null> {
  let capability = await loadSandboxCapability({ force: true });
  for (let left = attempts - 1; left > 0; left--) {
    if (capability === null || !capabilityPending(capability)) break;
    await new Promise((resolve) => setTimeout(resolve, delayMs));
    capability = await loadSandboxCapability({ force: true });
  }
  return capability;
}

export function cachedSandboxCapability(): SandboxCapability | null {
  return cached?.value ?? null;
}

export function forgetSandboxCapability(): void {
  cached = null;
  inFlight = null;
  latestRequest++;
  notifyCapabilityChange();
}

async function checkedSetup(
  res: Response,
  fallbackMessage: string,
): Promise<SandboxSetupJob> {
  if (!res.ok) {
    throw new Error(await readFastApiError(res, fallbackMessage));
  }
  return setupJobFromApi((await res.json()) as ApiSetupJob);
}

export async function startSandboxSetup(
  operation: SandboxSetupOperation,
  { consentDaclFallback = false }: { consentDaclFallback?: boolean },
  fallbackMessage: string,
): Promise<SandboxSetupJob> {
  const res = await authFetch(SETUP_ROUTE, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      operation,
      // biome-ignore lint/style/useNamingConvention: API schema
      consent_dacl_fallback: consentDaclFallback,
    }),
  });
  return checkedSetup(res, fallbackMessage);
}

export async function loadSandboxSetup(
  fallbackMessage: string,
): Promise<SandboxSetupJob> {
  const res = await authFetch(SETUP_ROUTE);
  return checkedSetup(res, fallbackMessage);
}
