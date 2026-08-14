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
// Pre-backend installs kept the record here; still read as a fallback so an
// upgrade does not forget the model.
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
        // The pre-backend v1 reader rejects entries without a numeric loadedAt,
        // and an older bundle or still-open tab shares this key.
        loadedAt,
        // True until the backend PUT for this record confirms: a differing
        // pending shadow may be newer than whatever the GET returns.
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
          // A local record the backend has not seen: a load whose PUT was
          // dropped at teardown (pendingSync), or one written by a still-open
          // pre-upgrade bundle that cannot stamp the marker or call the API at
          // all -- only the timestamps order loads across surfaces. Re-sync it
          // with its original load time even when the model identity matches
          // the backend record: the backend timestamp must advance, or an
          // older dropped write on another surface would later outrank it.
          // An unstamped backend record (pre-loaded_at writer) gives no such
          // order, so only a pending shadow -- this surface's own dropped
          // write -- outranks it; a plain stale shadow must not clobber it.
          recordLastLocalModelLoad({
            ...legacy.record,
            loadedAt: legacy.loadedAt,
          });
          return legacy.record;
        }
        if (legacy?.pendingSync) {
          // The backend record is at least as new: the shadow lost. Adopt the
          // backend copy and clear the marker.
          writeLegacyRecord(record, false, backendLoadedAt ?? Date.now());
        }
        return record;
      }
    }
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
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
  // Shadow write first, synchronously: a fetch still pending at document
  // teardown is dropped without running either callback, and the pre-backend
  // record was this surface's only memory of the load. It also covers the
  // pre-route backend that answers 404 without rejecting.
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
      // The server translates loaded_at into its own clock frame using this
      // (skew = server_now - client_now), so slow or fast local clocks cannot
      // strand or freeze the shared record.
      // biome-ignore lint/style/useNamingConvention: API schema
      client_now: Date.now(),
    }),
  })
    .then(async (res) => {
      if (!res.ok) {
        return;
      }
      // Adopt the server's answer: it may have clamped a future-dated stamp or
      // ignored this write as stale in favor of a newer one, and the shadow
      // must mirror the stored record or the next read would re-reconcile.
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
          // Shadow stamps live in this clock's frame: translate the server's
          // answer back before storing it next to locally stamped values.
          serverLoadedAt -= body.server_now - Date.now();
        }
      } catch {
        // Pre-loaded_at backend or opaque response: fall back to our stamp.
      }
      // Clear only this write's pending marker: a newer load may have replaced
      // the shadow while the PUT was in flight -- including a reload of the
      // same model, which identity alone cannot distinguish, so the timestamp
      // must match too or a slow older response would demote the newer shadow.
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
