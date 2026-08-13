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

// Describes the binary, which only changes on an install or an update, so one read
// per session is plenty. Shared so several mounted rows do not each pay the probe.
let inFlightCatalog: Promise<LlamaFlagCatalog | null> | null = null;
let cachedCatalog: LlamaFlagCatalog | null = null;

/**
 * Read the flag catalogue, or null when this backend has no such route.
 *
 * Null and `probeOk: false` mean the same thing to the caller (cannot verify), but
 * they are kept apart so an older backend is not reported as a broken llama.cpp.
 */
export function loadLlamaFlagCatalog(): Promise<LlamaFlagCatalog | null> {
  if (cachedCatalog) {
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
      return catalog;
    } catch {
      return null;
    } finally {
      inFlightCatalog = null;
    }
  })();
  return inFlightCatalog;
}
