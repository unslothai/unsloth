// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { Plugin } from "vite";

// Vite's preload nonce lookup walks the whole document per import; the meta can only sit in head.
const NONCE_LOOKUP = /document\.querySelector\((["'`])meta\[property=csp-nonce\]\1\)/g;

export function scopePreloadNonceLookup(code: string): string {
  return code.replace(NONCE_LOOKUP, "document.head.querySelector($1meta[property=csp-nonce]$1)");
}

export function preloadNonceInHead(): Plugin {
  return {
    name: "preload-nonce-in-head",
    apply: "build",
    renderChunk(code) {
      if (!code.includes("csp-nonce")) return null;
      const next = scopePreloadNonceLookup(code);
      return next === code ? null : { code: next, map: null };
    },
  };
}
