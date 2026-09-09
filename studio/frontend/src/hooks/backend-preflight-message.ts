// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Preflight reported a stale install; say which kind, since the fixes differ. */

/// The install is fine, the folder it must run from is not reachable.
/// Mirrors managed.rs.
export const WORKING_DIRECTORY_UNAVAILABLE = "working_directory_unavailable";

/// The folder is reachable; a path setting the user wrote is not resolvable.
/// Mirrors managed.rs.
export const PATH_SETTING_UNRESOLVABLE = "path_setting_unresolvable";

/// The managed llama.cpp runtime is installed but missing files. Mirrors the
/// reasons installed_runtime_health() returns in studio/install_llama_prebuilt.py.
export const LLAMA_RUNTIME_REASONS = [
  "llama_runtime_dir_missing",
  "llama_runtime_payload_incomplete",
  "llama_runtime_binaries_missing",
  "llama_runtime_incomplete",
];

export function preflightStaleMessage(
  disposition: string,
  reason: string | null,
): string {
  // The backend appends the setting it could not preserve, as `reason:NAME`.
  const [kind, setting] = (reason ?? "").split(":", 2);
  // Not an install problem: the home folder is unreachable and updating needs the
  // same folder. The roaming-profile cause is Windows-only but the reason reaches
  // every platform, so it is offered rather than asserted.
  if (kind === WORKING_DIRECTORY_UNAVAILABLE) {
    const cause =
      typeof navigator !== "undefined" && /Win/i.test(navigator.platform ?? "")
        ? " This usually means a network or roaming profile is not available yet."
        : "";
    return `Unsloth cannot reach your user folder, so it has nowhere to run from.${cause} Reconnect and try again.`;
  }
  // One of Unsloth's own path settings names somewhere unresolvable, so the value
  // is the fix, not an update.
  if (kind === PATH_SETTING_UNRESOLVABLE) {
    const which = setting ? `${setting} points` : "One of Unsloth's folder settings points";
    return `${which} somewhere that cannot be resolved, so Unsloth has nowhere safe to run from. Set it to a full path, such as D:\\unsloth-cache, and try again.`;
  }
  // The install is current, some of its files are gone, so "too old" sends people
  // to an update that reports they are up to date. Name the usual cause: a
  // reinstall into the same quarantine needs an exclusion, not a retry.
  if (LLAMA_RUNTIME_REASONS.includes(kind)) {
    return "Unsloth's llama.cpp runtime is missing files, which usually means security software quarantined them. Run `unsloth studio update` to reinstall it, and allow the folder in your antivirus if it happens again.";
  }
  if (disposition === "owned_stale") {
    return "Desktop-owned Unsloth backend is too old for this desktop app. Run `unsloth studio update`, then restart Unsloth.";
  }
  return "Managed Unsloth install is too old. Run `unsloth studio update`.";
}
