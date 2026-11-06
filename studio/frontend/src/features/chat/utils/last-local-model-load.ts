// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";

export type LastLocalModelKind = "gguf" | "model";

const PATH_LIKE_ID_RE = /^(?:[/~]|[A-Za-z]:[\\/]|\\\\)/;

function isPathLikeId(id: string): boolean {
  return PATH_LIKE_ID_RE.test(id);
}

export type LastLocalModelLoad = {
  id: string;
  kind: LastLocalModelKind;
  ggufVariant: string | null;
};

const API_PATH = "/api/settings/last-local-model";
// Legacy pre-backend key; still read so an upgrade does not forget the model.
const LEGACY_STORAGE_KEY = "unsloth.last-local-model-load.v1";

function isLastLocalModelKind(value: unknown): value is LastLocalModelKind {
  return value === "gguf" || value === "model";
}

function toRecord(input: {
  id?: unknown;
  kind?: unknown;
  ggufVariant?: unknown;
}): LastLocalModelLoad | null {
  if (typeof input.id !== "string" || !isLastLocalModelKind(input.kind)) {
    return null;
  }
  const id = input.id.trim();
  const ggufVariant =
    typeof input.ggufVariant === "string"
      ? input.ggufVariant.trim() || null
      : null;
  if (!id) {
    return null;
  }
  if (input.kind === "gguf" && !ggufVariant && !isPathLikeId(id)) {
    return null;
  }
  return { id, kind: input.kind, ggufVariant };
}

function sameRecord(a: LastLocalModelLoad, b: LastLocalModelLoad): boolean {
  return a.id === b.id && a.kind === b.kind && a.ggufVariant === b.ggufVariant;
}

function writeLegacyRecord(
  record: LastLocalModelLoad,
  pendingSync: boolean,
  loadedAt: number,
): void {
  try {
    localStorage.setItem(
      LEGACY_STORAGE_KEY,
      JSON.stringify({
        id: record.id,
        kind: record.kind,
        ggufVariant: record.ggufVariant,
        // Old bundles reject entries without a numeric loadedAt.
        loadedAt,
        pendingSync,
      }),
    );
  } catch {
    // Storage unavailable (private mode, quota): best effort only.
  }
}

type LegacyEntry = {
  record: LastLocalModelLoad;
  pendingSync: boolean;
  loadedAt: number | null;
};

function readLegacyEntry(): LegacyEntry | null {
  try {
    const raw = localStorage.getItem(LEGACY_STORAGE_KEY);
    if (!raw) {
      return null;
    }
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    const record = toRecord(parsed);
    if (!record) {
      return null;
    }
    return {
      record,
      pendingSync: parsed.pendingSync === true,
      loadedAt: typeof parsed.loadedAt === "number" ? parsed.loadedAt : null,
    };
  } catch {
    return null;
  }
}

export async function readLastLocalModelLoad(
  signal?: AbortSignal,
): Promise<LastLocalModelLoad | null> {
  try {
    const res = await authFetch(API_PATH, { signal });
    if (res.ok) {
      const data = (await res.json()) as {
        id?: unknown;
        kind?: unknown;
        // biome-ignore lint/style/useNamingConvention: API schema
        gguf_variant?: unknown;
        // biome-ignore lint/style/useNamingConvention: API schema
        loaded_at?: unknown;
        // biome-ignore lint/style/useNamingConvention: API schema
        server_now?: unknown;
      };
      const record = toRecord({
        id: data.id,
        kind: data.kind,
        ggufVariant: data.gguf_variant,
      });
      if (record) {
        const legacy = readLegacyEntry();
        let backendLoadedAt =
          typeof data.loaded_at === "number" ? data.loaded_at : null;
        if (backendLoadedAt !== null && typeof data.server_now === "number") {
          // Shadow stamps live in this clock's frame: compare like with like.
          backendLoadedAt -= data.server_now - Date.now();
        }
        if (
          legacy &&
          legacy.loadedAt !== null &&
          (backendLoadedAt === null
            ? legacy.pendingSync
            : legacy.loadedAt > backendLoadedAt)
        ) {
          // Re-sync an unseen local record with its original stamp; only a pending shadow outranks an unstamped one.
          recordLastLocalModelLoad({
            ...legacy.record,
            loadedAt: legacy.loadedAt,
          });
          return legacy.record;
        }
        if (legacy?.pendingSync) {
          writeLegacyRecord(record, false, backendLoadedAt ?? Date.now());
        }
        return record;
      }
    }
  } catch (err) {
    // Name, not instanceof: the retry wrapper surfaces aborts as a plain Error.
    if ((err as { name?: string } | null)?.name === "AbortError") {
      throw err;
    }
  }
  return readLegacyEntry()?.record ?? null;
}

export function recordLastLocalModelLoad(input: {
  id: string;
  kind: LastLocalModelKind;
  ggufVariant?: string | null;
  loadedAt?: number;
}): void {
  const record = toRecord(input);
  if (!record) {
    return;
  }
  const loadedAt =
    typeof input.loadedAt === "number" ? input.loadedAt : Date.now();
  // Write the shadow synchronously first: a pending fetch is dropped at teardown.
  writeLegacyRecord(record, true, loadedAt);
  authFetch(API_PATH, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      id: record.id,
      kind: record.kind,
      // biome-ignore lint/style/useNamingConvention: API schema
      gguf_variant: record.ggufVariant,
      // biome-ignore lint/style/useNamingConvention: API schema
      loaded_at: loadedAt,
      // biome-ignore lint/style/useNamingConvention: API schema
      client_now: Date.now(),
    }),
  })
    .then(async (res) => {
      if (!res.ok) {
        return;
      }
      // The server may clamp or reject this write; mirror what is stored.
      let serverRecord: LastLocalModelLoad | null = null;
      let serverLoadedAt: number | null = null;
      try {
        const body = (await res.json()) as {
          id?: unknown;
          kind?: unknown;
          // biome-ignore lint/style/useNamingConvention: API schema
          gguf_variant?: unknown;
          // biome-ignore lint/style/useNamingConvention: API schema
          loaded_at?: unknown;
          // biome-ignore lint/style/useNamingConvention: API schema
          server_now?: unknown;
        };
        serverRecord = toRecord({
          id: body.id,
          kind: body.kind,
          ggufVariant: body.gguf_variant,
        });
        serverLoadedAt =
          typeof body.loaded_at === "number" ? body.loaded_at : null;
        if (serverLoadedAt !== null && typeof body.server_now === "number") {
          serverLoadedAt -= body.server_now - Date.now();
        }
      } catch {
        // Pre-loaded_at backend or opaque response: fall back to our stamp.
      }
      // Clear only this write's marker: the stamp must match, since a newer load may have replaced it.
      const legacy = readLegacyEntry();
      if (
        !legacy?.pendingSync ||
        !sameRecord(legacy.record, record) ||
        legacy.loadedAt !== loadedAt
      ) {
        return;
      }
      if (serverRecord && !sameRecord(serverRecord, record)) {
        writeLegacyRecord(serverRecord, false, serverLoadedAt ?? Date.now());
      } else {
        writeLegacyRecord(record, false, serverLoadedAt ?? loadedAt);
      }
    })
    .catch(() => {
      // Best effort; the read path reconciles the pending shadow next launch.
    });
}
