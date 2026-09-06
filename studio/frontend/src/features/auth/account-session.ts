// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useSyncExternalStore } from "react";
import {
  ensureLoginMode,
  getLoginMode,
  subscribeLoginMode,
} from "./login-client";
import {
  AUTH_SESSION_CLEARED_EVENT,
  AUTH_SESSION_STORED_EVENT,
  getAuthToken,
} from "./session";

/** Display policy only; the server validates the token and enforces owner access. */
export function sessionAccount(
  token: string | null,
): { username: string; isOwner: boolean } | null {
  if (!token) return null;
  try {
    const payload = JSON.parse(
      atob(token.split(".")[1].replace(/-/g, "+").replace(/_/g, "/")),
    );
    if (typeof payload.sub !== "string") return null;
    return {
      username: payload.sub,
      isOwner: payload.role
        ? payload.role === "owner"
        : payload.sub === "unsloth",
    };
  } catch {
    return null;
  }
}

function subscribeSession(listener: () => void): () => void {
  const events = [
    AUTH_SESSION_STORED_EVENT,
    AUTH_SESSION_CLEARED_EVENT,
    "storage",
  ];
  for (const event of events) window.addEventListener(event, listener);
  return () => {
    for (const event of events) window.removeEventListener(event, listener);
  };
}
export function useIsAccountOwner(): boolean {
  return useSyncExternalStore(
    subscribeSession,
    () => sessionAccount(getAuthToken())?.isOwner ?? false,
    () => false,
  );
}
export function useLoginMode() {
  const mode = useSyncExternalStore(
    subscribeLoginMode,
    getLoginMode,
    () => "single" as const,
  );
  useEffect(ensureLoginMode, []);
  return mode;
}
