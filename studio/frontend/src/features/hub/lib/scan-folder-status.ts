// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Copy for a scan folder the backend could not read.
 *
 * A folder Unsloth is denied looks exactly like an empty one in the model list,
 * so the row has to say which it is and where to fix it.
 */

export type ScanFolderStatus =
  | "ok"
  | "permission_denied"
  | "missing"
  | "unreadable"
  | "partial";

export interface ScanFolderStatusCopy {
  title: string;
  hint: string;
}

/** Rough host detection: only picks which settings screen to name. */
function hostPlatform(userAgent: string): "mac" | "windows" | "other" {
  if (/Mac|iPhone|iPad/i.test(userAgent)) return "mac";
  if (/Win/i.test(userAgent)) return "windows";
  return "other";
}

function permissionHint(userAgent: string): string {
  switch (hostPlatform(userAgent)) {
    case "mac":
      return "Grant access in System Settings > Privacy & Security > Files and Folders, then reopen this dialog.";
    case "windows":
      return "Check the folder's security permissions, or allow Unsloth in Controlled Folder Access, then reopen this dialog.";
    default:
      return "Check the folder's permissions, then reopen this dialog.";
  }
}

export function scanFolderStatusCopy(
  status: ScanFolderStatus | undefined,
  userAgent: string = typeof navigator === "undefined" ? "" : navigator.userAgent,
): ScanFolderStatusCopy | null {
  switch (status) {
    case "permission_denied":
      return {
        title: "Unsloth is not allowed to read this folder",
        hint: permissionHint(userAgent),
      };
    case "partial":
      return {
        title: "Some models in this folder could not be read",
        hint: permissionHint(userAgent),
      };
    case "missing":
      return {
        title: "This folder is no longer there",
        hint: "It was moved, renamed, or is on a drive that is not connected.",
      };
    case "unreadable":
      return {
        title: "This folder could not be read",
        hint: "The drive may be disconnected or failing. Check it, then reopen this dialog.",
      };
    default:
      return null;
  }
}
