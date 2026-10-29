// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/// Mirrors WORKING_DIRECTORY_UNAVAILABLE in studio/src-tauri/src/preflight/managed.rs.
export const WORKING_DIRECTORY_UNAVAILABLE = "working_directory_unavailable";

/// Mirrors PATH_SETTING_UNRESOLVABLE in studio/src-tauri/src/preflight/managed.rs.
export const PATH_SETTING_UNRESOLVABLE = "path_setting_unresolvable";

/// Mirrors MANAGED_ENVIRONMENT_BUSY in studio/src-tauri/src/preflight/managed.rs.
export const MANAGED_ENVIRONMENT_BUSY = "managed_environment_busy";

/// Mirrors MANAGED_ENVIRONMENT_UPDATING in studio/src-tauri/src/preflight/managed.rs.
export const MANAGED_ENVIRONMENT_UPDATING = "managed_environment_updating";

/// Mirrors installed_runtime_health() reasons in studio/install_llama_prebuilt.py.
export const LLAMA_RUNTIME_REASONS = [
  "llama_runtime_dir_missing",
  "llama_runtime_payload_incomplete",
  "llama_runtime_binaries_missing",
  "llama_runtime_incomplete",
];

export function isLlamaRuntimeReason(reason: string | null): boolean {
  return LLAMA_RUNTIME_REASONS.includes((reason ?? "").split(":", 1)[0]);
}

function llamaRuntimeFolder(): string {
  return typeof navigator !== "undefined" && /Win/i.test(navigator.platform ?? "")
    ? "%USERPROFILE%\\.unsloth\\llama.cpp"
    : "~/.unsloth/llama.cpp";
}

export function runtimeRepairFailureMessage(error: string): string {
  return `Unsloth could not repair its llama.cpp runtime. Antivirus may be blocking the download or removing files as they are installed. Allow the runtime folder, usually ${llamaRuntimeFolder()}, in your antivirus, or check your connection, then retry. Repair error: ${error}`;
}

export function runtimeRepairRecurrenceMessage(): string {
  return `Unsloth's llama.cpp runtime is missing files again soon after a repair. Antivirus may be removing them. Allow the runtime folder, usually ${llamaRuntimeFolder()}, in your antivirus, then press Retry to reinstall it.`;
}

export function preflightStaleMessage(
  disposition: string,
  reason: string | null,
): string {
  // The backend appends `reason:NAME` for the setting it could not preserve.
  const [kind, setting] = (reason ?? "").split(":", 2);
  // The home folder is unreachable (probed on every platform), so the roaming-profile cause is
  // offered, not asserted.
  if (kind === WORKING_DIRECTORY_UNAVAILABLE) {
    const cause =
      typeof navigator !== "undefined" && /Win/i.test(navigator.platform ?? "")
        ? " This usually means a network or roaming profile is not available yet."
        : "";
    return `Unsloth cannot reach your user folder, so it has nowhere to run from.${cause} Reconnect and try again.`;
  }
  // One of Unsloth's own path settings is unresolvable; the value is the fix.
  if (kind === PATH_SETTING_UNRESOLVABLE) {
    const which = setting ? `${setting} points` : "One of Unsloth's folder settings points";
    return `${which} somewhere that cannot be resolved, so Unsloth has nowhere safe to run from. Set it to a full path, such as D:\\unsloth-cache, and try again.`;
  }
  // The install is current but files are gone (usually quarantine), so "too old" would mislead.
  if (isLlamaRuntimeReason(kind)) {
    // Names the default root (UNSLOTH_LLAMA_CPP_PATH can move it), hence "usually".
    return `Unsloth's llama.cpp runtime is missing files, which usually means security software quarantined them. Run \`unsloth studio update\` to reinstall it, and allow the folder it installs into, usually ${llamaRuntimeFolder()}, in your antivirus if it happens again.`;
  }
  if (disposition === "owned_stale") {
    return "Desktop-owned Unsloth backend is too old for this desktop app. Run `unsloth studio update`, then restart Unsloth.";
  }
  return "Managed Unsloth install is too old. Run `unsloth studio update`.";
}
