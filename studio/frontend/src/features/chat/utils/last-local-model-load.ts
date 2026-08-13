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
      };
      const record = toRecord({
        id: data.id,
        kind: data.kind,
        ggufVariant: data.gguf_variant,
      });
      if (record) {
        const legacy = readLegacyEntry();
        const backendLoadedAt =
          typeof data.loaded_at === "number" ? data.loaded_at : null;
        if (
          legacy?.pendingSync &&
          legacy.loadedAt !== null &&
          (backendLoadedAt === null || legacy.loadedAt > backendLoadedAt)
        ) {
          // A load whose PUT was dropped at teardown, and the backend has seen
          // nothing newer from any surface since (pendingSync alone proves the
          // write was dropped, not that it is the latest load -- only the
          // backend timestamp orders loads across surfaces). Re-sync it with
          // its original load time even when the model identity matches the
          // backend record: the backend timestamp must advance, or an older
          // dropped write on another surface would later outrank this load.
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
    }),
  })
    .then((res) => {
      if (!res.ok) {
        return;
      }
      // Clear only this write's pending marker: a newer load may have replaced
      // the shadow while the PUT was in flight -- including a reload of the
      // same model, which identity alone cannot distinguish, so the timestamp
      // must match too or a slow older response would demote the newer shadow.
      const legacy = readLegacyEntry();
      if (
        legacy?.pendingSync &&
        sameRecord(legacy.record, record) &&
        legacy.loadedAt === loadedAt
      ) {
        writeLegacyRecord(record, false, loadedAt);
      }
    })
    .catch(() => {
      // Best effort; the read path reconciles the pending shadow next launch.
    });
}
