// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Files that run code when opened always ask before downloading, whatever a site's remembered answer or
// the Ask before downloading setting.

import policy from "./dangerous-file-types.json" with { type: "json" };

const DANGEROUS = new Set<string>(policy.extensions);

/** Whether `name` runs code when opened, judged as the OS will: Windows drops trailing dots and spaces. */
export function isDangerousDownload(name: string): boolean {
  const base = name.slice(Math.max(name.lastIndexOf("/"), name.lastIndexOf("\\")) + 1).replace(/[. ]+$/, "");
  const dot = base.lastIndexOf(".");
  if (dot < 0) return false;
  // A bare `.exe` still runs as one. Windows compares names upper-cased, so `.mſi` (long s) is `.MSI`.
  const extension = base.slice(dot + 1);
  return DANGEROUS.has(extension.toLowerCase()) || DANGEROUS.has(extension.toUpperCase().toLowerCase());
}
