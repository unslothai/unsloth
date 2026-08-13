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

function writeLegacyRecord(record: LastLocalModelLoad): void {
  try {
    localStorage.setItem(
      LEGACY_STORAGE_KEY,
      JSON.stringify({
        id: record.id,
        kind: record.kind,
        ggufVariant: record.ggufVariant,
      }),
    );
  } catch {
    // Storage unavailable (private mode, quota): best effort only.
  }
}

function readLegacyRecord(): LastLocalModelLoad | null {
  try {
    const raw = localStorage.getItem(LEGACY_STORAGE_KEY);
    if (!raw) {
      return null;
    }
    return toRecord(JSON.parse(raw) as Record<string, unknown>);
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
      };
      const record = toRecord({
        id: data.id,
        kind: data.kind,
        ggufVariant: data.gguf_variant,
      });
      if (record) {
        return record;
      }
    }
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw err;
    }
    // Unreachable settings API: fall back to the legacy record.
  }
  return readLegacyRecord();
}

export function recordLastLocalModelLoad(input: {
  id: string;
  kind: LastLocalModelKind;
  ggufVariant?: string | null;
}): void {
  const record = toRecord(input);
  if (!record) {
    return;
  }
  // Shadow write first, synchronously: a fetch still pending at document
  // teardown is dropped without running either callback, and the pre-backend
  // record was this surface's only memory of the load. It also covers the
  // pre-route backend that answers 404 without rejecting.
  writeLegacyRecord(record);
  authFetch(API_PATH, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      id: record.id,
      kind: record.kind,
      // biome-ignore lint/style/useNamingConvention: API schema
      gguf_variant: record.ggufVariant,
    }),
  }).catch(() => {
    // Best effort; the read path falls back to the legacy record.
  });
}
