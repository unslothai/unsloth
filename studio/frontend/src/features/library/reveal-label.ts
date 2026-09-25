// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


const LABELS = {
  finder: "library.reveal.finder",
  explorer: "library.reveal.explorer",
  files: "library.reveal.files",
} as const;

export function isLoopbackHost(hostname: string): boolean {
  const host = hostname.toLowerCase().replace(/^\[(.*)\]$/, "$1").replace(/\.$/, "");
  if (host === "localhost" || host.endsWith(".localhost")) return true;
  if (/^127(\.\d{1,3}){3}$/.test(host) || host === "0.0.0.0") return true;
  if (host === "::1" || host === "::" || host === "0:0:0:0:0:0:0:1") return true;
  return /^::ffff:(127(\.\d{1,3}){3}|7f[0-9a-f]{2}:[0-9a-f]{1,4})$/.test(host);
}

export function revealLabelFor(
  fileManager: keyof typeof LABELS | null | undefined,
  deviceType: string,
): (typeof LABELS)[keyof typeof LABELS] | null {
  if (fileManager === null) return null;
  if (fileManager) return LABELS[fileManager];
  if (deviceType === "mac") return LABELS.finder;
  return deviceType === "windows" ? LABELS.explorer : LABELS.files;
}
