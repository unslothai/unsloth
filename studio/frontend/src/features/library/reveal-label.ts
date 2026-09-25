// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Kept free of app imports, so the rules can be tested on their own.

// Each platform's own wording for the command, as message keys: the caller translates them.
const LABELS = {
  finder: "library.reveal.finder",
  explorer: "library.reveal.explorer",
  files: "library.reveal.files",
} as const;

/** This machine however the URL spells it: localhost and its subdomains, 127.0.0.0/8, ::1 (bare,
 *  bracketed or IPv4-mapped), and 0.0.0.0 / ::, which browsers reach as this machine. */
export function isLoopbackHost(hostname: string): boolean {
  const host = hostname.toLowerCase().replace(/^\[(.*)\]$/, "$1").replace(/\.$/, "");
  if (host === "localhost" || host.endsWith(".localhost")) return true;
  if (/^127(\.\d{1,3}){3}$/.test(host) || host === "0.0.0.0") return true;
  if (host === "::1" || host === "::" || host === "0:0:0:0:0:0:0:1") return true;
  return /^::ffff:(127(\.\d{1,3}){3}|7f[0-9a-f]{2}:[0-9a-f]{1,4})$/.test(host);
}

/** The Reveal label's key for the file manager the server reports, or null where it would open
 *  nothing anyone here sees. An older server reports nothing, and its platform decides. */
export function revealLabelFor(
  fileManager: keyof typeof LABELS | null | undefined,
  deviceType: string,
): (typeof LABELS)[keyof typeof LABELS] | null {
  if (fileManager === null) return null;
  if (fileManager) return LABELS[fileManager];
  if (deviceType === "mac") return LABELS.finder;
  return deviceType === "windows" ? LABELS.explorer : LABELS.files;
}
