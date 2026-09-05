// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { backendLabel, limitedBackendLabel } from "./tool-isolation-labels";

export type ToolExecutionMode = "os_isolation_required" | "limited" | "full";

/** Outbound network for an OS-isolated launch: nothing, or the backend's fixed host allowlist
 *  through its local proxy. Only Required mode can enforce either. */
export type ToolNetworkPolicy = "deny" | "allowlist";

export const TOOL_NETWORK_POLICIES: readonly ToolNetworkPolicy[] = [
  "deny",
  "allowlist",
];

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
  available: boolean;
  qualified: boolean;
  limitations: string[];
  /** Network policies the Required-mode backend can enforce; always contains "deny". */
  network_policies: ToolNetworkPolicy[];
  /** Hosts an "allowlist" launch may reach. Empty when the host offers only "deny". */
  network_allowlist: string[];
  /** How Limited runs here when it is more than the software safeguards (Windows restricted
   *  token), else null. */
  limited_backend: string | null;
  limited_profile_id: string | null;
  limited_limitations: string[];
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
    const limitedBackend = limitedBackendLabel(
      capability?.limited_backend ?? null,
    );
    if (limitedBackend) {
      return {
        state: "limited",
        label: `Limited · ${limitedBackend}`,
        description:
          "Writes outside the sandbox directory are refused by the restricted token; each execution record says whether it applied. Reads, the network and other processes are not isolated.",
      };
    }
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
      label: `Protected · ${backendLabel(capability.backend, capability.environment, capability.profile_id)}`,
      description: "Python and Terminal use qualified OS isolation.",
    };
  }
  if (capability?.protection_state === "preview") {
    return {
      state: "preview",
      label: `Preview OS isolation · ${backendLabel(capability.backend, capability.environment, capability.profile_id)}`,
      description:
        "Python and Terminal use a preview sandbox whose live enforcement probe passed.",
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
    available: value.available === true,
    qualified: value.qualified === true,
    limitations: stringList(value.limitations),
    // Additive fields: a backend that predates the network proxy or the Windows
    // restricted token simply omits them, and the UI then offers neither.
    network_policies: networkPolicyList(value.network_policies),
    network_allowlist: stringList(value.network_allowlist),
    limited_backend:
      typeof value.limited_backend === "string" && value.limited_backend
        ? value.limited_backend
        : null,
    limited_profile_id:
      typeof value.limited_profile_id === "string" && value.limited_profile_id
        ? value.limited_profile_id
        : null,
    limited_limitations: stringList(value.limited_limitations),
  };
}

function stringList(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string")
    : [];
}

function networkPolicyList(value: unknown): ToolNetworkPolicy[] {
  const known = stringList(value).filter((item): item is ToolNetworkPolicy =>
    (TOOL_NETWORK_POLICIES as readonly string[]).includes(item),
  );
  // "deny" is always enforceable; an allowlist is offered only when the backend says so.
  return known.includes("deny") ? known : ["deny", ...known];
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
