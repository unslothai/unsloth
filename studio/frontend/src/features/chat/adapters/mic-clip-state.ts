// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A mic clip is sized to the audio room left when it starts; other audio added meanwhile would take that room.
let recording = false;

export function setMicClipRecording(value: boolean): void {
  recording = value;
}

export function micClipRecording(): boolean {
  return recording;
}
