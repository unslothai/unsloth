// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// One microphone consent flow for every recorder in the app.
//
// The desktop build answers WebView2's microphone requests from this store, so the
// browser's own permission bubble never appears. The in-app dialog asks once and
// remembers the answer; the composer, system-wide voice typing and the settings
// page all share it.

import { invoke } from "@tauri-apps/api/core";
import { create } from "zustand";

export type MicrophonePermission = "granted" | "denied" | "unknown";

/** Detect at call time: a cached module must not freeze the test's window stub. */
function isTauriNow(): boolean {
  if (typeof window === "undefined") return false;
  return (
    "__TAURI__" in window ||
    "__TAURI_INTERNALS__" in window ||
    window.location?.protocol === "tauri:"
  );
}

export type MicrophonePermissionReason =
  | "voice-typing"
  | "dictation"
  | "audio"
  | "settings";

interface MicrophonePermissionState {
  permission: MicrophonePermission;
  loaded: boolean;
  dialogOpen: boolean;
  reason: MicrophonePermissionReason;
  load: () => Promise<void>;
  request: (reason?: MicrophonePermissionReason) => Promise<boolean>;
  askAgain: (reason?: MicrophonePermissionReason) => Promise<boolean>;
  allow: () => Promise<void>;
  deny: () => Promise<void>;
  reset: () => Promise<void>;
}

let pending: ((allowed: boolean) => void) | null = null;
let loadPromise: Promise<void> | null = null;

function normalize(value: unknown): MicrophonePermission {
  return value === "granted" || value === "denied" ? value : "unknown";
}

async function readBrowserPermission(): Promise<MicrophonePermission> {
  try {
    const status = await navigator.permissions?.query({
      name: "microphone" as PermissionName,
    });
    return normalize(status?.state);
  } catch {
    return "unknown";
  }
}

async function focusMainWindow(): Promise<void> {
  if (!isTauriNow()) return;
  try {
    const { getCurrentWindow } = await import("@tauri-apps/api/window");
    await getCurrentWindow().setFocus();
  } catch {
    // Focus is best effort; the dialog still opens in the main window.
  }
}

function resolvePending(allowed: boolean): void {
  const resolve = pending;
  pending = null;
  resolve?.(allowed);
}

export const useMicrophonePermissionStore = create<MicrophonePermissionState>(
  (set, get) => ({
    permission: "unknown",
    loaded: false,
    dialogOpen: false,
    reason: "dictation",
    load: async () => {
      if (get().loaded) return;
      loadPromise ??= (async () => {
        const permission = isTauriNow()
          ? await invoke<string>("microphone_permission_state").catch(
              () => "unknown",
            )
          : await readBrowserPermission();
        set({ permission: normalize(permission), loaded: true });
      })();
      await loadPromise;
    },
    request: async (reason = "dictation") => {
      // A browser tab has no app-level permission to store; getUserMedia is the
      // prompt and the browser owns the answer.
      if (!isTauriNow()) return true;
      await get().load();
      if (get().permission === "granted") return true;
      // A saved deny is a decision, not a question. Nagging on every dictation
      // would be worse than the browser bubble this replaces; the settings page
      // and the blocked notice both offer a deliberate "ask again".
      if (get().permission === "denied") return false;
      if (pending) {
        // A second recorder asked while the dialog was open. Both callers get the
        // same answer instead of stacking dialogs.
        return new Promise<boolean>((resolve) => {
          const previous = pending;
          pending = (allowed) => {
            previous?.(allowed);
            resolve(allowed);
          };
        });
      }
      await focusMainWindow();
      return new Promise<boolean>((resolve) => {
        pending = resolve;
        set({ dialogOpen: true, reason });
      });
    },
    askAgain: async (reason = "dictation") => {
      // Only an explicit request clears a saved answer, so the dialog cannot
      // reappear on its own after the user said no.
      await get().reset();
      return get().request(reason);
    },
    allow: async () => {
      if (isTauriNow()) {
        await invoke("set_microphone_permission", { state: "granted" }).catch(
          (error) => {
            console.warn("Could not store the microphone permission:", error);
          },
        );
      }
      set({ permission: "granted", dialogOpen: false, loaded: true });
      resolvePending(true);
    },
    deny: async () => {
      if (isTauriNow()) {
        await invoke("set_microphone_permission", { state: "denied" }).catch(
          (error) => {
            console.warn("Could not store the microphone permission:", error);
          },
        );
      }
      set({ permission: "denied", dialogOpen: false, loaded: true });
      resolvePending(false);
    },
    reset: async () => {
      if (isTauriNow()) {
        await invoke("reset_microphone_permission").catch((error) => {
          console.warn("Could not reset the microphone permission:", error);
        });
      }
      set({ permission: "unknown", dialogOpen: false, loaded: true });
      resolvePending(false);
    },
  }),
);

/** Ensure the app has consent before a recorder asks the engine for the mic. */
export async function ensureMicrophonePermission(
  reason: MicrophonePermissionReason = "dictation",
  options: { askAgain?: boolean } = {},
): Promise<boolean> {
  if (!isTauriNow()) return true;
  const store = useMicrophonePermissionStore.getState();
  return options.askAgain ? store.askAgain(reason) : store.request(reason);
}

/** Read the current answer without opening the dialog. */
export async function readMicrophonePermission(): Promise<MicrophonePermission> {
  if (!isTauriNow()) return readBrowserPermission();
  await useMicrophonePermissionStore.getState().load();
  return useMicrophonePermissionStore.getState().permission;
}
