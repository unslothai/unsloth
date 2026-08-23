// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Shared by the Hub catalog and every model picker. Keep the legacy key stable. */
export const MODELS_FIT_ON_DEVICE_ONLY_KEY =
  "unsloth_models_fit_on_device_only";

type Listener = (value: boolean) => void;

function readStoredPreference(): boolean {
  if (typeof window === "undefined") {
    return false;
  }
  try {
    return (
      window.localStorage.getItem(MODELS_FIT_ON_DEVICE_ONLY_KEY) === "true"
    );
  } catch {
    return false;
  }
}

let fitOnDeviceOnly = readStoredPreference();
const listeners = new Set<Listener>();

export function getFitOnDeviceOnlyPreference(): boolean {
  return fitOnDeviceOnly;
}

export function subscribeFitOnDeviceOnlyPreference(
  listener: Listener,
): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function setFitOnDeviceOnlyPreference(value: boolean): void {
  const changed = value !== fitOnDeviceOnly;
  fitOnDeviceOnly = value;
  if (typeof window !== "undefined") {
    try {
      window.localStorage.setItem(MODELS_FIT_ON_DEVICE_ONLY_KEY, String(value));
    } catch {
      // Keep the in-memory preference when storage is unavailable.
    }
  }
  if (!changed) {
    return;
  }
  for (const listener of [...listeners]) {
    listener(value);
  }
}
