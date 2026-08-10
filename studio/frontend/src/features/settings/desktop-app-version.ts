// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

type VersionReader = () => Promise<string>;

async function readTauriAppVersion(): Promise<string> {
  const { getVersion } = await import("@tauri-apps/api/app");
  return getVersion();
}

// Callers gate this on isTauri: null means desktop without a readable version,
// and never reaching the loader is what keeps the row off browser builds.
export async function loadDesktopAppVersion(
  readVersion: VersionReader = readTauriAppVersion,
): Promise<string | null> {
  try {
    const version = await readVersion();
    return version.trim() || null;
  } catch (error) {
    console.warn(
      "Tauri app version read failed; showing it as unavailable",
      error,
    );
    return null;
  }
}
