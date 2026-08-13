// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";

/** What the installed llama-server accepts, for checking pass-through args. */
export type LlamaFlagCatalog = {
  /** Flag name -> its help text. Empty when the probe failed. */
  flags: Record<string, string>;
  /** Flags Unsloth manages; the load refuses these outright. */
  managed: ReadonlySet<string>;
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
// Bumped by every invalidation. A read that started before the binary changed is
// describing the old one, so it must neither be cached nor handed to a caller that
// asked after the change: without this the stale promise is reused by the next call
// and writes the previous backend's flags back for the whole TTL.
let catalogGeneration = 0;

let inFlightManaged: Promise<ReadonlySet<string> | null> | null = null;
let cachedManaged: ReadonlySet<string> | null = null;

/** Drop the cache, for a caller that knows the binary just changed. */
export function invalidateLlamaFlagCatalog(): void {
  cachedCatalog = null;
  cachedAt = 0;
  catalogGeneration += 1;
  // Dropped as well as cleared: a request already on the wire answers for the
  // binary that has just been replaced.
  inFlightCatalog = null;
  // Not the denylist: that is Unsloth's own list and no binary changes it.
}

/**
 * Just the flags Unsloth refuses, without the `--help` probe behind the catalogue.
 *
 * The panel sanitizes a stored list with this before turning it into an explicit
 * request, and that must not wait on a cold probe (up to ten seconds), or a flag
 * denied since the list was saved stays in the request for as long as it runs.
 * Cached for the session: unlike the flag map, it describes this build of Studio.
 */
export function loadManagedLlamaFlags(): Promise<ReadonlySet<string> | null> {
  if (cachedManaged) {
    return Promise.resolve(cachedManaged);
  }
  if (cachedCatalog) {
    return Promise.resolve(cachedCatalog.managed);
  }
  inFlightManaged ??= (async () => {
    try {
      const res = await authFetch(
        "/api/inference/llama-flags?managed_only=true",
      );
      if (!res.ok) {
        return null;
      }
      const body = (await res.json()) as ApiLlamaFlagCatalog;
      // An older backend answers the route without the parameter and returns its
      // full catalogue, which carries the same list.
      cachedManaged = new Set(body.managed ?? []);
      return cachedManaged;
    } catch {
      return null;
    } finally {
      inFlightManaged = null;
    }
  })();
  return inFlightManaged;
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
  const generation = catalogGeneration;
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
      if (generation !== catalogGeneration) {
        // The binary changed while this was in flight: answer the caller with
        // "cannot verify" rather than the old build's flags, and cache nothing.
        return null;
      }
      cachedCatalog = catalog;
      cachedAt = Date.now();
      return catalog;
    } catch {
      return null;
    } finally {
      // Only if it is still ours: an invalidation may have cleared it already, and
      // a later call may have started its own.
      if (generation === catalogGeneration) {
        inFlightCatalog = null;
      }
    }
  })();
  return inFlightCatalog;
}
