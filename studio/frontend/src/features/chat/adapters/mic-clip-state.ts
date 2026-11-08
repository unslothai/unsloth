// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A mic clip is sized to the audio room left when it starts; other audio added meanwhile would take that room.
// Owned per session, so a stale session's late cleanup cannot clear a newer recording's hold.
let owner: object | null = null;

export function holdMicClipRoom(session: object): void {
  owner = session;
}

export function releaseMicClipRoom(session: object): void {
  if (owner === session) owner = null;
}

export function micClipRecording(): boolean {
  return owner !== null;
}
