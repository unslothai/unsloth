// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Desktop only: a file tab opened in another app, as ChatGPT's Open menu does. The app writes the
// tab's bytes to a local copy first (browser_open_with.rs) and hands back an id for it.

import { NATIVE_FILE_NAME_HEADER, encodeNativeFilename } from "@/lib/native-files";

export type LocalCopy = { id: string; path: string };
export type OpenWithApp = { path: string; name: string; icon: string | null; default: boolean };

async function invoke<T>(command: string, args?: Record<string, unknown>): Promise<T> {
  const core = await import("@tauri-apps/api/core");
  return core.invoke<T>(command, args);
}

// By blob: a tab's bytes never change, so its copy is written once however often it is opened.
const copies = new WeakMap<Blob, Map<string, Promise<LocalCopy>>>();

export function localCopy(blob: Blob, name: string): Promise<LocalCopy> {
  const byName = copies.get(blob) ?? new Map<string, Promise<LocalCopy>>();
  copies.set(blob, byName);
  const known = byName.get(name);
  if (known) return known;
  const pending = blob.arrayBuffer().then(async (bytes) => {
    const core = await import("@tauri-apps/api/core");
    return core.invoke<LocalCopy>("browser_file_local_copy", new Uint8Array(bytes), {
      headers: { [NATIVE_FILE_NAME_HEADER]: encodeNativeFilename(name) },
    });
  });
  byName.set(name, pending);
  // A failure is not kept, so the next try writes it again.
  pending.catch(() => byName.delete(name));
  return pending;
}

export function openWithApps(id: string): Promise<OpenWithApp[]> {
  return invoke<OpenWithApp[]>("browser_file_apps", { id });
}

/** With its default app when `app` is omitted. */
export function openLocalCopy(id: string, app?: string): Promise<void> {
  return invoke<void>("browser_file_open", { id, with: app ?? null });
}

export function revealLocalCopy(id: string): Promise<void> {
  return invoke<void>("browser_file_reveal", { id });
}
