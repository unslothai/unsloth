// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** what the page runtime's `inspect` says an element would do (see browser_agent.js). */
export type SensitiveKind =
  | "password"
  | "credentials"
  | "payment"
  | "purchase"
  | "send"
  | "delete";

export type BrowserActionKind =
  | "navigate"
  | "click"
  | "type"
  | "select"
  | "press";

export type ApprovalNeed =
  | { ask: false }
  | { ask: true; reason: "ask-mode" | "sensitive" | "new-site" };

const SENSITIVE_KINDS: ReadonlySet<string> = new Set([
  "password",
  "credentials",
  "payment",
  "purchase",
  "send",
  "delete",
]);

export function asSensitiveKind(value: unknown): SensitiveKind | null {
  return typeof value === "string" && SENSITIVE_KINDS.has(value)
    ? (value as SensitiveKind)
    : null;
}

/** sensitive actions ask even on an allowed site; unrecognised modes ask like ask mode; auto skips navigation, since opening a page is not acting on it. */
export function approvalNeeded({
  mode,
  kind,
  sensitive,
  siteAllowed,
}: {
  mode: string;
  kind: BrowserActionKind;
  sensitive: SensitiveKind | null;
  siteAllowed: boolean;
}): ApprovalNeed {
  if (mode === "off" || mode === "full") return { ask: false };
  if (sensitive) return { ask: true, reason: "sensitive" };
  if (siteAllowed) return { ask: false };
  if (mode !== "auto") return { ask: true, reason: "ask-mode" };
  return kind === "navigate"
    ? { ask: false }
    : { ask: true, reason: "new-site" };
}

export function sensitiveDetail(
  sensitive: SensitiveKind | null,
): string | null {
  switch (sensitive) {
    case "password":
    case "credentials":
      return "This signs in with a password.";
    case "payment":
      return "This form takes payment details.";
    case "purchase":
      return "This looks like it completes a purchase.";
    case "send":
      return "This looks like it sends or posts something.";
    case "delete":
      return "This looks like it deletes something.";
    default:
      return null;
  }
}
