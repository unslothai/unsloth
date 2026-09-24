// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Kept free of app imports, so the rules can be tested on their own.

type FileManager = "finder" | "explorer" | "files" | null;

// Each platform's own wording for the command, as message keys: the caller translates them.
const LABELS = {
  finder: "library.reveal.finder",
  explorer: "library.reveal.explorer",
  files: "library.reveal.files",
} as const;

export type RevealLabelKey = (typeof LABELS)[keyof typeof LABELS];

/**
 * Hosts that are this machine, however the URL spells it: localhost and its subdomains, all of
 * 127.0.0.0/8, ::1 (bare, bracketed or IPv4-mapped), and 0.0.0.0 / ::, which browsers connect to
 * as this machine when Studio is bound to every interface.
 */
export function isLoopbackHost(hostname: string): boolean {
  const host = hostname.toLowerCase().replace(/^\[(.*)\]$/, "$1").replace(/\.$/, "");
  if (host === "localhost" || host.endsWith(".localhost")) return true;
  if (/^127(\.\d{1,3}){3}$/.test(host) || host === "0.0.0.0") return true;
  if (host === "::1" || host === "::" || host === "0:0:0:0:0:0:0:1") return true;
  return /^::ffff:(127(\.\d{1,3}){3}|7f[0-9a-f]{2}:[0-9a-f]{1,4})$/.test(host);
}

/**
 * The message key of the Reveal command's label for the server's host, or null where it would open
 * nothing anyone here sees. `fileManager` is what the server reports; an older server reports nothing, and its
 * platform decides.
 */
export function revealLabelFor(
  fileManager: FileManager | undefined,
  deviceType: string,
): RevealLabelKey | null {
  if (fileManager === null) return null;
  if (fileManager) return LABELS[fileManager];
  if (deviceType === "mac") return LABELS.finder;
  return deviceType === "windows" ? LABELS.explorer : LABELS.files;
}
