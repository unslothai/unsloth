// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";

export type LlamaFlagCatalog = {
  flags: Record<string, string>;
  managed: ReadonlySet<string>;
  switches: ReadonlySet<string>;
  maxBytes: number;
  /** Windows quoted-command char budget, 0 elsewhere; quoting can double backslashes. */
  windowsCommandBudget: number;
  /** llama-server aborts on a batch below the slots it serves. */
  defaultParallelSlots: number;
  /** Build without --kv-unified serves one slot, so an explicit Slots value must not size the batch floor. */
  parallelSlotsClamped: boolean;
  /** False when `--help` could not be read: an unverifiable flag is not a typo. */
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

// Short TTL: an update or backend switch replaces the binary while the tab stays open.
const CATALOG_TTL_MS = 60_000;
let inFlightCatalog: Promise<LlamaFlagCatalog | null> | null = null;
let cachedCatalog: LlamaFlagCatalog | null = null;
let cachedAt = 0;
// Bumped per invalidation so a read of the old binary is neither cached nor handed out.
let catalogGeneration = 0;

export type LlamaManagedFlags = {
  managed: ReadonlySet<string>;
  maxBytes: number;
  windowsCommandBudget: number;
  defaultParallelSlots: number;
  parallelSlotsClamped: boolean;
};

let inFlightManaged: Promise<LlamaManagedFlags | null> | null = null;
let cachedManaged: LlamaManagedFlags | null = null;

const catalogListeners = new Set<() => void>();

/** Updating llama.cpp replaces the binary while the panel stays mounted. */
export function subscribeLlamaFlagCatalog(listener: () => void): () => void {
  catalogListeners.add(listener);
  return () => {
    catalogListeners.delete(listener);
  };
}

export function invalidateLlamaFlagCatalog(): void {
  cachedCatalog = null;
  cachedAt = 0;
  catalogGeneration += 1;
  // A request already in flight answers for the replaced binary.
  inFlightCatalog = null;
  // It carries defaultParallelSlots, which depends on the binary.
  cachedManaged = null;
  inFlightManaged = null;
  for (const listener of catalogListeners) {
    listener();
  }
}

/** No --help probe (up to 10s); cached for the session since the denylist is Unsloth's own. */
export function loadManagedLlamaFlags(): Promise<LlamaManagedFlags | null> {
  if (cachedManaged) {
    return Promise.resolve(cachedManaged);
  }
  if (cachedCatalog) {
    return Promise.resolve(cachedCatalog);
  }
  // Generation-checked as in the full catalogue: defaultParallelSlots depends on the binary.
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
      const managed: LlamaManagedFlags = {
        managed: new Set(body.managed ?? []),
        maxBytes: body.max_bytes ?? 0,
        windowsCommandBudget: body.windows_command_budget ?? 0,
        // 0 on an older backend means unknown, leaving the editor's hard floor of 2 in charge.
        defaultParallelSlots: body.default_parallel_slots ?? 0,
        parallelSlotsClamped: Boolean(body.parallel_slots_clamped),
      };
      if (generation !== catalogGeneration) {
        // Binary changed mid-flight: answer "cannot verify" and cache nothing.
        return null;
      }
      cachedManaged = managed;
      return cachedManaged;
    } catch {
      return null;
    } finally {
      // Only if still ours: an invalidation or a later call may have replaced it.
      if (generation === catalogGeneration) {
        inFlightManaged = null;
      }
    }
  })();
  return inFlightManaged;
}

/** null (older backend) and `probeOk: false` (broken probe) both mean cannot verify. */
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
        // 0 means no limit of the backend's own; the editor default applies.
        maxBytes: body.max_bytes ?? 0,
        windowsCommandBudget: body.windows_command_budget ?? 0,
        // 0 on an older backend means unknown, leaving the editor's hard floor of 2 in charge.
        defaultParallelSlots: body.default_parallel_slots ?? 0,
        parallelSlotsClamped: Boolean(body.parallel_slots_clamped),
        probeOk: Boolean(body.probe_ok),
      };
      if (generation !== catalogGeneration) {
        // Binary changed mid-flight: answer "cannot verify" and cache nothing.
        return null;
      }
      cachedCatalog = catalog;
      cachedAt = Date.now();
      return catalog;
    } catch {
      return null;
    } finally {
      // Only if still ours: an invalidation or a later call may have replaced it.
      if (generation === catalogGeneration) {
        inFlightCatalog = null;
      }
    }
  })();
  return inFlightCatalog;
}
