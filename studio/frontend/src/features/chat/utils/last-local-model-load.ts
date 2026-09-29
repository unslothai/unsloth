// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";

export type LastLocalModelKind = "gguf" | "model";

const PATH_LIKE_ID_RE = /^(?:[/~]|[A-Za-z]:[\\/]|\\\\)/;

/** A filesystem target rather than a Hub repo id. */
function isPathLikeId(id: string): boolean {
  return PATH_LIKE_ID_RE.test(id);
}

export type LastLocalModelLoad = {
  id: string;
  kind: LastLocalModelKind;
  ggufVariant: string | null;
};

const API_PATH = "/api/settings/last-local-model";
// Pre-backend installs kept the record here; still read so an upgrade does not forget the model.
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
  // A quant-less cached repo names no file; a local .gguf path is the file.
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
        // True until this record's PUT confirms; a pending shadow may be newer.
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
          // A local record the backend has not seen: a PUT dropped at teardown (pendingSync) or a
          // pre-upgrade bundle's write. Re-sync it with its original stamp, even on an identity match, so
          // the backend stamp advances past older dropped writes elsewhere. An unstamped backend record
          // gives no order, so only a pending shadow may outrank it.
          recordLastLocalModelLoad({
            ...legacy.record,
            loadedAt: legacy.loadedAt,
          });
          return legacy.record;
        }
        if (legacy?.pendingSync) {
          // The backend record is at least as new: adopt it, clear the marker.
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
    // Unreachable settings API: fall back to the legacy record.
  }
  return readLegacyEntry()?.record ?? null;
}

export function recordLastLocalModelLoad(input: {
  id: string;
  kind: LastLocalModelKind;
  ggufVariant?: string | null;
  // Reconcile re-issues keep the original load time; fresh loads stamp now.
  loadedAt?: number;
}): void {
  const record = toRecord(input);
  if (!record) {
    return;
  }
  const loadedAt =
    typeof input.loadedAt === "number" ? input.loadedAt : Date.now();
  // Shadow write first, synchronously: a fetch pending at teardown is dropped without running
  // either callback. Also covers a pre-route backend's 404.
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
      // The server shifts loaded_at by (server_now - client_now).
      // biome-ignore lint/style/useNamingConvention: API schema
      client_now: Date.now(),
    }),
  })
    .then(async (res) => {
      if (!res.ok) {
        return;
      }
      // Adopt the server's answer: it may have clamped or rejected this write, and the shadow must
      // mirror what is stored.
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
          // Shadow stamps live in this clock's frame: translate back first.
          serverLoadedAt -= body.server_now - Date.now();
        }
      } catch {
        // Pre-loaded_at backend or opaque response: fall back to our stamp.
      }
      // Clear only this write's marker: a newer load may have replaced the shadow mid-flight, even a
      // reload of the same model, so the stamp must match too.
      const legacy = readLegacyEntry();
      if (
        !legacy?.pendingSync ||
        !sameRecord(legacy.record, record) ||
        legacy.loadedAt !== loadedAt
      ) {
        return;
      }
      if (serverRecord && !sameRecord(serverRecord, record)) {
        // The server kept a newer record from another surface: ours lost.
        writeLegacyRecord(serverRecord, false, serverLoadedAt ?? Date.now());
      } else {
        writeLegacyRecord(record, false, serverLoadedAt ?? loadedAt);
      }
    })
    .catch(() => {
      // Best effort; the read path reconciles the pending shadow next launch.
    });
}
