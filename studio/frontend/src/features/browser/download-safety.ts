// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Files that run code when opened: the panel asks before saving one, and the desktop app keeps one
// under a neutral name until the reader says to keep it. src-tauri reads the same list. Loaded on the
// first download, not at startup.

import policy from "./dangerous-file-types.json" with { type: "json" };

const DANGEROUS = new Set<string>(policy.extensions);

/** Whether `name` runs code when opened, judged as the OS will: Windows drops trailing dots and spaces. */
export function isDangerousDownload(name: string): boolean {
  const base = name.slice(Math.max(name.lastIndexOf("/"), name.lastIndexOf("\\")) + 1).replace(/[. ]+$/, "");
  const dot = base.lastIndexOf(".");
  // A bare `.exe` still runs as one.
  return dot >= 0 && DANGEROUS.has(base.slice(dot + 1).toLowerCase());
}

/** Bidi controls out, so `x\u202efdp.exe` can't read as `xexe.pdf` where it is shown or saved. */
export function safeDownloadName(name: string): string {
  return name.replace(/[\u200e\u200f\u202a-\u202e\u2066-\u2069]/g, "_");
}
