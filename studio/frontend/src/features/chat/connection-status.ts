// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  ProviderAuthKind,
  ProviderAuthStatus,
  ProviderTestErrorKind,
} from "./api/providers-api";

export interface ConnectionCheck {
  ok: boolean;
  message: string;
  kind?: ProviderTestErrorKind | null;
  checkedAt: number;
}

export type ConnectionStatusTone = "ok" | "warn" | "error" | "muted";

export interface ConnectionStatus {
  tone: ConnectionStatusTone;
  label: string;
  detail?: string;
  /** The provider's own wording, for a tooltip; `detail` is the short form a row has room for. */
  full?: string;
}

const FAILED_CHECK_LABELS: Record<ProviderTestErrorKind, string> = {
  auth: "Key rejected",
  billing: "No credit",
  rate_limit: "Rate limited",
  not_found: "No models endpoint",
  unreachable: "Unreachable",
  timeout: "Timed out",
  invalid_url: "Invalid base URL",
  other: "Check failed",
};

const FAILED_CHECK_HINTS: Partial<Record<ProviderTestErrorKind, string>> = {
  auth: "Check the API key.",
  billing: "Add credit to the account.",
  rate_limit: "Try again shortly.",
  not_found: "Check the base URL.",
  unreachable: "Is the server running?",
  timeout: "The server took too long.",
  invalid_url: "Check the base URL.",
};

const FAILED_CHECK_PREFIX = /^Connection failed[:.]\s*/;

export function checkFailureDetail(message: string): string {
  return message.replace(FAILED_CHECK_PREFIX, "").trim();
}

export function formatCheckedAgo(checkedAt: number, now: number): string {
  const seconds = Math.max(0, Math.round((now - checkedAt) / 1000));
  if (seconds < 45) return "just now";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

export function connectionStatus(
  provider: {
    authKind?: ProviderAuthKind;
    authStatus?: ProviderAuthStatus;
    hasApiKey?: boolean;
    models: readonly string[];
  },
  options: {
    enabled: boolean;
    /** Self-hosted servers and browser-held legacy keys work without a saved key. */
    keyOptional: boolean;
    checking?: boolean;
    check?: ConnectionCheck;
    now?: number;
  },
): ConnectionStatus {
  if (!options.enabled) return { tone: "muted", label: "Off" };
  if (options.checking) return { tone: "muted", label: "Checking" };
  if (provider.authKind === "chatgpt_oauth") {
    if (provider.authStatus === "reauthorization_required") {
      return {
        tone: "warn",
        label: "Sign in again",
        detail: "The sign-in expired.",
      };
    }
    if (provider.authStatus !== "connected") {
      return {
        tone: "warn",
        label: "Not signed in",
        detail: "Sign in with ChatGPT.",
      };
    }
  } else if (!provider.hasApiKey && !options.keyOptional) {
    return {
      tone: "warn",
      label: "Needs API key",
      detail: "Add a key.",
    };
  }
  const check = options.check;
  if (check && !check.ok) {
    return {
      tone: "error",
      label:
        FAILED_CHECK_LABELS[check.kind ?? "other"] ?? FAILED_CHECK_LABELS.other,
      detail:
        FAILED_CHECK_HINTS[check.kind ?? "other"] ??
        checkFailureDetail(check.message),
      full: checkFailureDetail(check.message),
    };
  }
  if (provider.models.length === 0) {
    return {
      tone: "warn",
      label: "No models enabled",
      detail: "Choose models.",
    };
  }
  if (check) {
    return {
      tone: "ok",
      label: "Connected",
      detail: `Checked ${formatCheckedAgo(check.checkedAt, options.now ?? Date.now())}`,
    };
  }
  return {
    tone: "muted",
    label: provider.authKind === "chatgpt_oauth" ? "Signed in" : "Ready",
  };
}

/** Suggestions the provider still lists; else every model of a short self-hosted list. */
export function suggestedModelIds(
  seeds: readonly string[],
  listed: readonly string[],
  limit = 6,
): string[] {
  const live = new Set(listed);
  const suggested = seeds.filter((id) => live.has(id)).slice(0, limit);
  if (suggested.length > 0) return suggested;
  return seeds.length === 0 && listed.length <= limit ? [...listed] : [];
}
