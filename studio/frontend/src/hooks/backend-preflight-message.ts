// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Preflight reported a stale install; say which kind, since the fixes differ. */

/// The install is fine, the folder it must run from is not reachable.
/// Mirrors WORKING_DIRECTORY_UNAVAILABLE in studio/src-tauri/src/preflight/managed.rs.
export const WORKING_DIRECTORY_UNAVAILABLE = "working_directory_unavailable";

export function preflightStaleMessage(
  disposition: string,
  reason: string | null,
): string {
  // Not an install problem: the home folder itself is unreachable. Telling this
  // user to update points them at a command that needs the same folder.
  //
  // The roaming-profile explanation is a Windows one, and this reason is
  // reachable on every platform: `home_dir_available()` is called ungated from
  // the preflight probe. On Linux or macOS the same symptom means an unmounted
  // home or a permissions problem, so the cause is only offered where it
  // applies rather than asserted everywhere.
  if (reason === WORKING_DIRECTORY_UNAVAILABLE) {
    const cause =
      typeof navigator !== "undefined" && /Win/i.test(navigator.platform ?? "")
        ? " This usually means a network or roaming profile is not available yet."
        : "";
    return `Unsloth cannot reach your user folder, so it has nowhere to run from.${cause} Reconnect and try again.`;
  }
  if (disposition === "owned_stale") {
    return "Desktop-owned Unsloth backend is too old for this desktop app. Run `unsloth studio update`, then restart Unsloth.";
  }
  return "Managed Unsloth install is too old. Run `unsloth studio update`.";
}
