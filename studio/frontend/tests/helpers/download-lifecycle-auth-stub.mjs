// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const AUTH_SESSION_CLEARED_EVENT = "unsloth:auth-session-cleared";
export const AUTH_SESSION_MARK_KEY = "unsloth_auth_session_mark";
export const AUTH_SESSION_STORED_EVENT = "unsloth:auth-session-stored";
export const AUTH_TOKEN_KEY = "unsloth_auth_token";

export function authFetch(input, init) {
  return globalThis.fetch(input, init);
}
