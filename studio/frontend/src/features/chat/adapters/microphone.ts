// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// One microphone acquisition path for dictation. The composer and system-wide
// voice typing both need the same saved-device preference, the same fallback
// when that device was unplugged, and the same explanation when the OS refuses
// access.

import { ensureMicrophonePermission } from "@/features/microphone";
import { isMissingDeviceError } from "./studio-web-speech-dictation-adapter";

const BASE_CONSTRAINTS: MediaTrackConstraints = {
  echoCancellation: true,
  noiseSuppression: true,
  channelCount: 1,
};

export function stopMediaStream(stream: MediaStream | null): void {
  for (const track of stream?.getTracks() ?? []) {
    track.stop();
  }
}

/** Open the saved microphone, falling back to the system default when it is
 *  gone. Throws the browser's error so the caller can distinguish a denied
 *  permission from a missing device. */
export async function openMicrophone(deviceId: string): Promise<MediaStream> {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("Microphone capture is not available in this browser.");
  }
  // The desktop app asks with its own dialog first; the native gate then answers
  // WebView2 from that choice, so the browser bubble never appears.
  if (!(await ensureMicrophonePermission("dictation"))) {
    throw new DOMException(
      "Microphone access was not granted.",
      "NotAllowedError",
    );
  }
  const wanted = deviceId.trim();
  if (wanted && wanted !== "default") {
    try {
      return await navigator.mediaDevices.getUserMedia({
        audio: { ...BASE_CONSTRAINTS, deviceId: { exact: wanted } },
      });
    } catch (error) {
      if (!isMissingDeviceError(error)) throw error;
      // The saved device is unplugged: the default is better than no dictation.
    }
  }
  return navigator.mediaDevices.getUserMedia({ audio: BASE_CONSTRAINTS });
}

export function microphoneErrorMessage(error: unknown): string {
  const name =
    error && typeof error === "object" && "name" in error
      ? (error as { name?: unknown }).name
      : undefined;
  switch (name) {
    case "NotAllowedError":
    case "SecurityError":
      return "Microphone access is blocked. Allow it in Voice settings and try again.";
    case "NotFoundError":
    case "OverconstrainedError":
      return "No microphone was found for dictation.";
    case "NotReadableError":
      return "The microphone is busy in another app.";
    default:
      return "Dictation could not access the microphone.";
  }
}
