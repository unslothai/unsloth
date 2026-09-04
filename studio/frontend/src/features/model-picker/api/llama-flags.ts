// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";

/** What the installed llama-server accepts, for checking pass-through args. */
export type LlamaFlagCatalog = {
  /** Flag name -> its help text. Empty when the probe failed. */
  flags: Record<string, string>;
  /** Flags Unsloth manages; the load refuses these outright. */
  managed: ReadonlySet<string>;
  /** Flags this build documents as taking no value ("--verbose", "--jinja"). */
  switches: ReadonlySet<string>;
  /** Size limit this host applies, which is smaller on Windows. */
  maxBytes: number;
  /**
   * Characters the quoted command may take on Windows, or 0 elsewhere. Mirrored
   * because the quoting can double a backslash-heavy value, so bytes alone do not
   * say whether the launch will fit.
   */
  windowsCommandBudget: number;
  /**
   * Slots a load gets when it names none, the server-wide --parallel. Needed to
   * judge a pass-through --batch-size: llama-server aborts on a batch below the
   * slots it serves, and with the Slots field blank this is what the launch uses.
   */
  defaultParallelSlots: number;
  /**
   * True when this build serves ONE slot however many are asked for (no
   * --kv-unified, so load_model falls back). defaultParallelSlots is already
   * effective, but an EXPLICIT Slots value chosen in this panel is not, and sizing
   * the batch floor from it refuses a --batch-size the backend accepts.
   */
  parallelSlotsClamped: boolean;
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
  switch_flags?: string[];
  // biome-ignore lint/style/useNamingConvention: API schema
  max_bytes?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  windows_command_budget?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  default_parallel_slots?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  parallel_slots_clamped?: boolean;
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

/** What a caller gets without paying for the --help probe. */
export type LlamaManagedFlags = {
  managed: ReadonlySet<string>;
  maxBytes: number;
  windowsCommandBudget: number;
  defaultParallelSlots: number;
  /**
   * True when this build serves ONE slot however many are asked for (no
   * --kv-unified, so load_model falls back). defaultParallelSlots is already
   * effective, but an EXPLICIT Slots value chosen in this panel is not, and sizing
   * the batch floor from it refuses a --batch-size the backend accepts.
   */
  parallelSlotsClamped: boolean;
};

let inFlightManaged: Promise<LlamaManagedFlags | null> | null = null;
let cachedManaged: LlamaManagedFlags | null = null;

const catalogListeners = new Set<() => void>();

/**
 * Told when the catalogue is invalidated, for a consumer holding one in state.
 *
 * Clearing the module cache is not enough for a row that already read it: updating
 * llama.cpp from the banner replaces the binary while the panel stays mounted, and
 * the row would go on judging arity against the old build's help until the section
 * is closed and reopened.
 */
export function subscribeLlamaFlagCatalog(listener: () => void): () => void {
  catalogListeners.add(listener);
  return () => {
    catalogListeners.delete(listener);
  };
}

/** Drop the cache, for a caller that knows the binary just changed. */
export function invalidateLlamaFlagCatalog(): void {
  cachedCatalog = null;
  cachedAt = 0;
  catalogGeneration += 1;
  // Dropped as well as cleared: a request already on the wire answers for the
  // binary that has just been replaced.
  inFlightCatalog = null;
  // The managed answer too. Its denylist is Unsloth's own and no binary changes it,
  // but it carries defaultParallelSlots beside it, and that is the EFFECTIVE count:
  // a build without --kv-unified serves one slot however many are configured. A tab
  // that had already fetched it went on sizing the hidden hydration check's batch
  // floor from the previous backend, letting "--batch-size 2" through on a build
  // that now serves four slots, or refusing it after a clamp to one.
  cachedManaged = null;
  inFlightManaged = null;
  for (const listener of catalogListeners) {
    listener();
  }
}

/**
 * Just the flags Unsloth refuses, without the `--help` probe behind the catalogue.
 *
 * The panel sanitizes a stored list with this before turning it into an explicit
 * request, and that must not wait on a cold probe (up to ten seconds), or a flag
 * denied since the list was saved stays in the request for as long as it runs.
 * Cached for the session: unlike the flag map, it describes this build of Unsloth.
 */
export function loadManagedLlamaFlags(): Promise<LlamaManagedFlags | null> {
  if (cachedManaged) {
    return Promise.resolve(cachedManaged);
  }
  if (cachedCatalog) {
    return Promise.resolve(cachedCatalog);
  }
  // Read before the request goes out, and checked before its answer is published,
  // exactly as the full catalogue does. The denylist in this answer is Unsloth's own
  // and no binary changes it, but defaultParallelSlots beside it is the EFFECTIVE
  // count, which depends on the probed binary: a request already on the wire when
  // llama.cpp is replaced would otherwise repopulate the cache the invalidation had
  // just cleared, and keep the old build's slot floor for the rest of the session.
  const generation = catalogGeneration;
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
      const managed: LlamaManagedFlags = {
        managed: new Set(body.managed ?? []),
        maxBytes: body.max_bytes ?? 0,
        windowsCommandBudget: body.windows_command_budget ?? 0,
        // 0 on a backend that predates the field, which reads as "not known" and
        // leaves the editor's own hard floor of 2 in charge.
        defaultParallelSlots: body.default_parallel_slots ?? 0,
        // Absent on a backend that predates the field, and false is the safe read:
        // the floor then follows the asked-for count, exactly as it did before.
        parallelSlotsClamped: Boolean(body.parallel_slots_clamped),
      };
      if (generation !== catalogGeneration) {
        // The binary changed while this was in flight: answer "cannot verify"
        // rather than the old build's limits, and cache nothing.
        return null;
      }
      cachedManaged = managed;
      return cachedManaged;
    } catch {
      return null;
    } finally {
      // Only if it is still ours: an invalidation may have cleared it already, and
      // a later call may have started its own.
      if (generation === catalogGeneration) {
        inFlightManaged = null;
      }
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
        switches: new Set(body.switch_flags ?? []),
        // An older backend sends neither, and 0 reads as "no limit of my own",
        // leaving the editor's own default in charge.
        maxBytes: body.max_bytes ?? 0,
        windowsCommandBudget: body.windows_command_budget ?? 0,
        // 0 on a backend that predates the field, which reads as "not known" and
        // leaves the editor's own hard floor of 2 in charge.
        defaultParallelSlots: body.default_parallel_slots ?? 0,
        parallelSlotsClamped: Boolean(body.parallel_slots_clamped),
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
