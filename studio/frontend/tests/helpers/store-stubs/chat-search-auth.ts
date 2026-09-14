// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Mirrors features/auth/session-events, which the real module re-exports.
export const AUTH_SESSION_CLEARED_EVENT = "unsloth:auth-session-cleared";
export const AUTH_SESSION_MARK_KEY = "unsloth_auth_session_mark";
// Not imported by the module under test: a test asserts a refresh of this key is ignored.
export const AUTH_TOKEN_KEY = "unsloth_auth_token";

/** The auth session epoch, driveable so a test can simulate a sign-out inside one page load. */
let epoch = 0;

export function getAuthSessionEpoch(): number {
  return epoch;
}

export function setAuthSessionEpochForTest(next: number): void {
  epoch = next;
}
