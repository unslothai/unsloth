// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const MAX_RUN_CONFIG_URL_LENGTH = 16_384;
export const nativeRunAddress = /^unsloth:\/\/run\/?(?:\?|$)/i;
export const runConfigHash = /^#run(?:\?|$)/;

export function isRunConfigLink(raw: string): boolean {
  if (raw.length > MAX_RUN_CONFIG_URL_LENGTH) {
    const fragment = raw.indexOf("#");
    return (
      nativeRunAddress.test(raw) ||
      (/^https?:\/\//i.test(raw) &&
        fragment >= 0 &&
        (raw.slice(fragment, fragment + 5) === "#run?" ||
          raw.slice(fragment) === "#run"))
    );
  }
  try {
    const url = new URL(raw);
    return (
      (url.protocol === "unsloth:" && url.hostname.toLowerCase() === "run") ||
      ((url.protocol === "http:" || url.protocol === "https:") &&
        runConfigHash.test(url.hash))
    );
  } catch {
    return false;
  }
}
