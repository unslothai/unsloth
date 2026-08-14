// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Preflight reported a stale install; say which kind, since the fixes differ. */

/// The install is fine, the folder it must run from is not reachable.
/// Mirrors WORKING_DIRECTORY_UNAVAILABLE in studio/src-tauri/src/preflight/managed.rs.
export const WORKING_DIRECTORY_UNAVAILABLE = "working_directory_unavailable";

/// The folder is reachable; a path setting the user wrote is not resolvable.
/// Mirrors PATH_SETTING_UNRESOLVABLE in studio/src-tauri/src/preflight/managed.rs.
export const PATH_SETTING_UNRESOLVABLE = "path_setting_unresolvable";

export function preflightStaleMessage(
  disposition: string,
  reason: string | null,
): string {
  // The backend appends the setting it could not preserve, as `reason:NAME`, so
  // the discriminator is the part before the colon and the name is the rest.
  const [kind, setting] = (reason ?? "").split(":", 2);
  // Not an install problem: the home folder itself is unreachable, and updating
  // needs the same folder. The roaming-profile cause is a Windows one, but this
  // reason reaches every platform (`home_dir_available()` is probed ungated), so
  // it is offered rather than asserted.
  if (kind === WORKING_DIRECTORY_UNAVAILABLE) {
    const cause =
      typeof navigator !== "undefined" && /Win/i.test(navigator.platform ?? "")
        ? " This usually means a network or roaming profile is not available yet."
        : "";
    return `Unsloth cannot reach your user folder, so it has nowhere to run from.${cause} Reconnect and try again.`;
  }
  // Also not an install problem, and not the folder either: one of Unsloth's own
  // path settings names somewhere unresolvable, so the value is the fix.
  if (kind === PATH_SETTING_UNRESOLVABLE) {
    const which = setting ? `${setting} points` : "One of Unsloth's folder settings points";
    return `${which} somewhere that cannot be resolved, so Unsloth has nowhere safe to run from. Set it to a full path, such as D:\\unsloth-cache, and try again.`;
  }
  if (disposition === "owned_stale") {
    return "Desktop-owned Unsloth backend is too old for this desktop app. Run `unsloth studio update`, then restart Unsloth.";
  }
  return "Managed Unsloth install is too old. Run `unsloth studio update`.";
}
