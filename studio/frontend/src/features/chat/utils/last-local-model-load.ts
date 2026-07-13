// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type LastLocalModelKind = "gguf" | "model";

export type LastLocalModelLoad = {
  id: string;
  kind: LastLocalModelKind;
  ggufVariant: string | null;
  loadedAt: number;
};

const STORAGE_KEY = "unsloth.last-local-model-load.v1";

function storage(): Storage | null {
  try {
    return typeof localStorage === "undefined" ? null : localStorage;
  } catch {
    return null;
  }
}

function isLastLocalModelKind(value: unknown): value is LastLocalModelKind {
  return value === "gguf" || value === "model";
}

export function readLastLocalModelLoad(): LastLocalModelLoad | null {
  try {
    const raw = storage()?.getItem(STORAGE_KEY);
    if (!raw) {
      return null;
    }
    const parsed = JSON.parse(raw) as Partial<LastLocalModelLoad>;
    if (
      typeof parsed.id !== "string" ||
      !parsed.id.trim() ||
      !isLastLocalModelKind(parsed.kind) ||
      typeof parsed.loadedAt !== "number"
    ) {
      return null;
    }
    if (
      parsed.kind === "gguf" &&
      (typeof parsed.ggufVariant !== "string" || !parsed.ggufVariant.trim())
    ) {
      return null;
    }
    return {
      id: parsed.id,
      kind: parsed.kind,
      ggufVariant:
        typeof parsed.ggufVariant === "string" ? parsed.ggufVariant : null,
      loadedAt: parsed.loadedAt,
    };
  } catch {
    return null;
  }
}

export function recordLastLocalModelLoad(input: {
  id: string;
  kind: LastLocalModelKind;
  ggufVariant?: string | null;
}): void {
  const id = input.id.trim();
  if (!id) {
    return;
  }
  const ggufVariant = input.ggufVariant?.trim() || null;
  if (input.kind === "gguf" && !ggufVariant) {
    return;
  }
  try {
    storage()?.setItem(
      STORAGE_KEY,
      JSON.stringify({
        id,
        kind: input.kind,
        ggufVariant: input.kind === "gguf" ? ggufVariant : null,
        loadedAt: Date.now(),
      } satisfies LastLocalModelLoad),
    );
  } catch {
    // Ignore disabled storage / quota errors; auto-load falls back to size order.
  }
}
