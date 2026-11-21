// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Module state, read by the recording bar after the session is gone: composer text cannot
 *  tell (a saved prompt may be inserted, a partial transcript still lands). */
let producedTranscript = false;
let producedClip: File | null = null;
let failed = false;

export function beginDictationSession(): void {
  producedTranscript = false;
  producedClip = null;
  failed = false;
}

export function markDictationClip(file: File): void {
  producedClip = file;
}

export function dictationProducedClip(): File | null {
  return producedClip;
}

export function markDictationTranscript(): void {
  producedTranscript = true;
}

export function dictationProducedTranscript(): boolean {
  return producedTranscript;
}

/** Text may be partial rather than absent, since both engines publish what did transcribe. */
export function markDictationFailed(): void {
  failed = true;
}

export function dictationFailed(): boolean {
  return failed;
}
