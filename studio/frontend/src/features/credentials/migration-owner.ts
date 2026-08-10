// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type LegacyCredentialOwnerAction = "claim" | "keep" | "ignore";

export function legacyCredentialOwnerAction(
  storedOwner: string | null,
  currentOwner: string,
): LegacyCredentialOwnerAction {
  if (!storedOwner) return "claim";
  return storedOwner === currentOwner ? "keep" : "ignore";
}

/** Read the authenticated subject only as a local ownership label; the backend validates it. */
export function authSubjectFromJwt(token: string): string | null {
  try {
    const payload = token.split(".")[1];
    if (!payload) return null;
    const normalized = payload.replace(/-/g, "+").replace(/_/g, "/");
    const padded = normalized.padEnd(
      normalized.length + ((4 - (normalized.length % 4)) % 4),
      "=",
    );
    if (typeof atob !== "function") return null;
    const decoded = atob(padded);
    const subject = (JSON.parse(decoded) as { sub?: unknown }).sub;
    return typeof subject === "string" && subject.length > 0 ? subject : null;
  } catch {
    return null;
  }
}
