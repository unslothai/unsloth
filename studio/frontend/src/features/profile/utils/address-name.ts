// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** What the app calls the user: nickname, first name, then the login. */
export function addressName(profile: { nickname: string; displayName: string }, login: string): string {
  return profile.nickname.trim() || profile.displayName.trim().split(/\s+/)[0] || login;
}
