// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";

export type ToolExecutionMode = "os_isolation_required" | "limited" | "full";

export type ToolIsolationProtectionState =
  | "protected"
  | "preview"
  | "unavailable";

/** Advisory host capability. The launch route revalidates it before exec. */
export type ToolIsolationCapability = {
  environment: string;
  backend: string | null;
  protection_state: ToolIsolationProtectionState;
  profile_id: string | null;
  probe_generation: string;
  environment_fingerprint: string;
  reason: string | null;
  remediation: string | null;
  retryable: boolean;
};

/** Opaque, backend-issued consent proof. Never persist this value. */
export type LimitedToolGrant = {
  grant: string;
  expires_at: string | number;
  probe_generation: string;
};

export type ToolIsolationPresentation = {
  state: "protected" | "preview" | "unavailable" | "limited" | "full";
  label: string;
  description: string;
};

function backendLabel(backend: string | null, environment: string): string {
  if (!backend?.toLowerCase().includes("bubblewrap")) {
    return backend || "OS sandbox";
  }
  const normalized = environment.toLowerCase();
  if (normalized.includes("wsl")) {
    return "Bubblewrap (WSL2)";
  }
  if (normalized.includes("colab")) {
    return "Bubblewrap (Colab)";
  }
  if (normalized.includes("container")) {
    return "Bubblewrap (Container)";
  }
  return "Bubblewrap";
}

export function isLimitedGrantCurrent(
  grant: LimitedToolGrant | null,
  capability: ToolIsolationCapability | null,
): grant is LimitedToolGrant {
  if (!(grant && capability)) {
    return false;
  }
  const expiresAt =
    typeof grant.expires_at === "number"
      ? grant.expires_at < 1_000_000_000_000
        ? grant.expires_at * 1000
        : grant.expires_at
      : Date.parse(grant.expires_at);
  return (
    grant.probe_generation === capability.probe_generation &&
    Number.isFinite(expiresAt) &&
    expiresAt > Date.now()
  );
}

export function toolIsolationPresentation(
  mode: ToolExecutionMode,
  capability: ToolIsolationCapability | null,
  grant: LimitedToolGrant | null = null,
): ToolIsolationPresentation {
  if (mode === "full") {
    return {
      state: "full",
      label: "Full access · security restrictions disabled",
      description: "Python and Terminal run with unrestricted host access.",
    };
  }
  if (mode === "limited" && isLimitedGrantCurrent(grant, capability)) {
    return {
      state: "limited",
      label: "Limited · no OS isolation",
      description:
        "Software safeguards remain active, but Limited is not an OS sandbox.",
    };
  }
  if (!capability) {
    return {
      state: "unavailable",
      label: "Checking OS isolation…",
      description: "Python and Terminal wait for a live capability check.",
    };
  }
  if (capability?.protection_state === "protected") {
    return {
      state: "protected",
      label: `Protected · ${backendLabel(capability.backend, capability.environment)}`,
      description: "Python and Terminal use qualified OS isolation.",
    };
  }
  if (capability?.protection_state === "preview") {
    return {
      state: "preview",
      label: `Preview OS isolation · ${backendLabel(capability.backend, capability.environment)}`,
      description:
        "Python and Terminal use an environment-qualified preview sandbox.",
    };
  }
  return {
    state: "unavailable",
    label: "OS isolation unavailable",
    description: "Python and Terminal are blocked until you choose a mode.",
  };
}

export function createToolIsolationUiSessionId(): string {
  if (typeof globalThis.crypto?.randomUUID === "function") {
    return globalThis.crypto.randomUUID();
  }
  const bytes = new Uint8Array(16);
  globalThis.crypto?.getRandomValues?.(bytes);
  if (bytes.some((value) => value !== 0)) {
    return Array.from(bytes, (value) =>
      value.toString(16).padStart(2, "0"),
    ).join("");
  }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

function responseError(body: unknown, fallback: string): string {
  if (body && typeof body === "object") {
    const record = body as Record<string, unknown>;
    for (const key of ["detail", "message", "error"]) {
      if (typeof record[key] === "string" && record[key]) {
        return record[key];
      }
    }
    if (record.detail && typeof record.detail === "object") {
      return responseError(record.detail, fallback);
    }
  }
  return fallback;
}

function parseCapability(body: unknown): ToolIsolationCapability {
  if (!body || typeof body !== "object") {
    throw new Error("Invalid tool-isolation capability response");
  }
  const value = body as Record<string, unknown>;
  if (
    typeof value.environment !== "string" ||
    !["protected", "preview", "unavailable"].includes(
      String(value.protection_state),
    ) ||
    typeof value.probe_generation !== "string" ||
    typeof value.environment_fingerprint !== "string"
  ) {
    throw new Error("Invalid tool-isolation capability response");
  }
  return {
    environment: value.environment,
    backend: typeof value.backend === "string" ? value.backend : null,
    protection_state: value.protection_state as ToolIsolationProtectionState,
    profile_id: typeof value.profile_id === "string" ? value.profile_id : null,
    probe_generation: value.probe_generation,
    environment_fingerprint: value.environment_fingerprint,
    reason: typeof value.reason === "string" ? value.reason : null,
    remediation:
      typeof value.remediation === "string" ? value.remediation : null,
    retryable: value.retryable === true,
  };
}

function parseGrant(body: unknown): LimitedToolGrant {
  if (!body || typeof body !== "object") {
    throw new Error("Invalid Limited grant response");
  }
  const value = body as Record<string, unknown>;
  if (
    typeof value.grant !== "string" ||
    !value.grant ||
    (typeof value.expires_at !== "string" &&
      typeof value.expires_at !== "number") ||
    typeof value.probe_generation !== "string"
  ) {
    throw new Error("Invalid Limited grant response");
  }
  return {
    grant: value.grant,
    expires_at: value.expires_at,
    probe_generation: value.probe_generation,
  };
}

let capabilityRequest: Promise<ToolIsolationCapability> | null = null;

export async function fetchToolIsolationCapability(): Promise<ToolIsolationCapability> {
  if (capabilityRequest) {
    return capabilityRequest;
  }
  capabilityRequest = (async () => {
    const response = await authFetch(
      "/api/inference/tool-isolation/capability",
    );
    const body = await response.json().catch(() => null);
    if (!response.ok) {
      throw new Error(
        responseError(body, `Capability check failed (${response.status})`),
      );
    }
    return parseCapability(body);
  })();
  try {
    return await capabilityRequest;
  } finally {
    capabilityRequest = null;
  }
}

export async function fetchLimitedToolGrant(
  uiSessionId: string,
  probeGeneration: string,
): Promise<LimitedToolGrant> {
  const response = await authFetch(
    "/api/inference/tool-isolation/limited-grant",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ui_session_id: uiSessionId,
        probe_generation: probeGeneration,
      }),
    },
  );
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(
      responseError(body, `Limited grant failed (${response.status})`),
    );
  }
  return parseGrant(body);
}
