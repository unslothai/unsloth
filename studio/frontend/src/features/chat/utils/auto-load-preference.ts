// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const CHAT_AUTO_LOAD_MODEL_KEY = "unsloth_chat_auto_load_model";
export const DEFAULT_CHAT_AUTO_LOAD_MODEL = true;

function canUseStorage(): boolean {
  return typeof window !== "undefined";
}

/** Whether sending without a selected model may load one from local storage. */
export function readAutoLoadModelPreference(): boolean {
  if (!canUseStorage()) {
    return DEFAULT_CHAT_AUTO_LOAD_MODEL;
  }
  try {
    const raw = window.localStorage.getItem(CHAT_AUTO_LOAD_MODEL_KEY);
    if (raw === "false") {
      return false;
    }
    if (raw === "true") {
      return true;
    }
  } catch {
    // Treat an unavailable browser store as the default so chat remains usable.
  }
  return DEFAULT_CHAT_AUTO_LOAD_MODEL;
}

export function saveAutoLoadModelPreference(enabled: boolean): void {
  if (!canUseStorage()) {
    return;
  }
  try {
    window.localStorage.setItem(CHAT_AUTO_LOAD_MODEL_KEY, String(enabled));
  } catch {
    // A read-only browser store should not prevent changing the in-memory toggle.
  }
}
