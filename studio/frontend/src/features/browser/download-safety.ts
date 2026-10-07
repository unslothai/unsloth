// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Files that run code when opened always ask before downloading, whatever a site's remembered answer or
// the Ask before downloading setting.

import policy from "./dangerous-file-types.json" with { type: "json" };

const DANGEROUS = new Set<string>(policy.extensions);

// native_file_dialogs.rs MAX_DOWNLOAD_NAME_BYTES.
const MAX_NAME_BYTES = 240;
const utf8 = new TextEncoder();
const byteLength = (text: string) => utf8.encode(text).length;
// Saved with a leading "_" (native_file_dialogs.rs WINDOWS_DEVICE_NAMES).
const DEVICE = /^(con|prn|aux|nul|com[1-9]|lpt[1-9])$/i;

/** The name the desktop app saves it under (native_file_dialogs.rs `safe_download_name`): past 240 bytes it
 *  keeps an extension of at most 32 bytes, else cuts the end, so a long tail can leave `.exe` last. */
function savedAs(name: string): string {
  const shift = DEVICE.test(name.split(".")[0].trimEnd()) ? 1 : 0;
  if (byteLength(name) + shift <= MAX_NAME_BYTES) return name;
  const dot = name.lastIndexOf(".");
  const extension = dot > 0 && byteLength(name.slice(dot)) <= 32 ? name.slice(dot) : "";
  const room = MAX_NAME_BYTES - shift - byteLength(extension);
  let stem = "";
  let used = 0;
  for (const char of name) {
    used += byteLength(char);
    if (used > room) break;
    stem += char;
  }
  return stem + extension;
}

function runsCode(name: string): boolean {
  const base = name.replace(/[. ]+$/, "");
  const dot = base.lastIndexOf(".");
  if (dot < 0) return false;
  // A bare `.exe` still runs as one. Windows compares names upper-cased, so `.m\u017fi` (long s) is `.MSI`.
  const extension = base.slice(dot + 1);
  return DANGEROUS.has(extension.toLowerCase()) || DANGEROUS.has(extension.toUpperCase().toLowerCase());
}

/** Whether `name` runs code when opened, judged as the OS will (Windows drops trailing dots and spaces),
 *  both as given and as the desktop app shortens it. */
export function isDangerousDownload(name: string): boolean {
  // As the desktop app cleans it before the cut: controls, bidi controls and reserved characters become
  // one-byte "_", trailing dots and spaces go.
  const flat = name
    .replace(/[\u0000-\u001f\u007f-\u009f\u061c\u200e\u200f\u202a-\u202e\u2066-\u2069/\\:*?"<>|]/g, "_")
    .replace(/[. ]+$/, "");
  return (
    runsCode(name.slice(Math.max(name.lastIndexOf("/"), name.lastIndexOf("\\")) + 1)) ||
    runsCode(savedAs(flat))
  );
}
