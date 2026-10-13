// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Windows Long Paths explained before the install (src-tauri/src/long_paths.rs).

export type LongPathsStatus = "enabled" | "disabled" | "unsupported";
export type EnableLongPathsOutcome = "enabled" | "declined" | "failed";

// Only a definite "disabled" prompts; an error installs as before.
export async function windowsLongPathsNeedPrompt(): Promise<boolean> {
  try {
    const { invoke } = await import("@tauri-apps/api/core");
    const status = await invoke<LongPathsStatus>("windows_long_paths_status");
    return status === "disabled";
  } catch {
    return false;
  }
}

export async function enableWindowsLongPaths(): Promise<EnableLongPathsOutcome> {
  try {
    const { invoke } = await import("@tauri-apps/api/core");
    return await invoke<EnableLongPathsOutcome>("enable_windows_long_paths");
  } catch {
    return "failed";
  }
}
