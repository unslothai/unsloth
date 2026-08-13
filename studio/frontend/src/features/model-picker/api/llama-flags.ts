// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";

/** What the installed llama-server accepts, for checking pass-through args. */
export type LlamaFlagCatalog = {
  /** Flag name -> its help text. Empty when the probe failed. */
  flags: Record<string, string>;
  /** Flags Unsloth manages; the load refuses these outright. */
  managed: Set<string>;
  /**
   * False when `--help` could not be read. Nothing may then be reported as a typo:
   * an unverifiable flag is not a wrong one.
   */
  probeOk: boolean;
};

type ApiLlamaFlagCatalog = {
  flags?: Record<string, string>;
  managed?: string[];
  // biome-ignore lint/style/useNamingConvention: API schema
  probe_ok?: boolean;
};

// Describes the binary, so it is worth caching: several rows can mount at once and
// none of them should pay a separate probe. Not for the whole session though. A
// llama.cpp update and a backend switch (CUDA to Vulkan, say) both replace the
// binary in place while the tab stays open, and the flag list moves with it, so a
// session-long cache would keep reporting a newly added flag as unknown. A minute
// is short enough that nobody meets that, and long enough that a panel being opened
// and closed does not re-ask. The backend keys its own cache on the binary's
// revision, so a repeat read is a dict lookup and not another --help.
const CATALOG_TTL_MS = 60_000;
let inFlightCatalog: Promise<LlamaFlagCatalog | null> | null = null;
let cachedCatalog: LlamaFlagCatalog | null = null;
let cachedAt = 0;

/** Drop the cache, for a caller that knows the binary just changed. */
export function invalidateLlamaFlagCatalog(): void {
  cachedCatalog = null;
  cachedAt = 0;
}

/**
 * Read the flag catalogue, or null when this backend has no such route.
 *
 * Null and `probeOk: false` mean the same thing to the caller (cannot verify), but
 * they are kept apart so an older backend is not reported as a broken llama.cpp.
 */
export function loadLlamaFlagCatalog(): Promise<LlamaFlagCatalog | null> {
  if (cachedCatalog && Date.now() - cachedAt < CATALOG_TTL_MS) {
    return Promise.resolve(cachedCatalog);
  }
  inFlightCatalog ??= (async () => {
    try {
      const res = await authFetch("/api/inference/llama-flags");
      if (!res.ok) {
        return null;
      }
      const body = (await res.json()) as ApiLlamaFlagCatalog;
      const catalog: LlamaFlagCatalog = {
        flags: body.flags ?? {},
        managed: new Set(body.managed ?? []),
        probeOk: Boolean(body.probe_ok),
      };
      cachedCatalog = catalog;
      cachedAt = Date.now();
      return catalog;
    } catch {
      return null;
    } finally {
      inFlightCatalog = null;
    }
  })();
  return inFlightCatalog;
}
