// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type DefaultChatModelKind = "gguf" | "model";

export type DefaultChatModel = {
  id: string;
  kind: DefaultChatModelKind;
  ggufVariant: string | null;
};

export const DEFAULT_CHAT_MODEL_STORAGE_KEY = "unsloth.default-chat-model.v1";
export const DEFAULT_CHAT_MODEL_STARTUP_KEY =
  "unsloth.default-chat-model.startup.v1";

function storage(): Storage | null {
  return typeof localStorage === "undefined" ? null : localStorage;
}

function parseRecord(value: unknown): DefaultChatModel | null {
  if (!value || typeof value !== "object") return null;
  const record = value as Record<string, unknown>;
  if (
    typeof record.id !== "string" ||
    !record.id.trim() ||
    (record.kind !== "gguf" && record.kind !== "model")
  ) {
    return null;
  }
  const ggufVariant =
    typeof record.ggufVariant === "string"
      ? record.ggufVariant.trim() || null
      : null;
  if (
    record.kind === "gguf" &&
    !ggufVariant &&
    !isPathLikeId(record.id.trim())
  ) {
    return null;
  }
  return {
    id: record.id.trim(),
    kind: record.kind,
    ggufVariant,
  };
}

function isPathLikeId(id: string): boolean {
  return /^(?:[/~]|[A-Za-z]:[\\/]|\\\\)/.test(id);
}

export function readDefaultChatModel(): DefaultChatModel | null {
  const store = storage();
  if (!store) return null;
  try {
    const raw = store.getItem(DEFAULT_CHAT_MODEL_STORAGE_KEY);
    return raw ? parseRecord(JSON.parse(raw)) : null;
  } catch {
    return null;
  }
}

export function setDefaultChatModel(model: DefaultChatModel): void {
  const store = storage();
  if (!store) return;
  const parsed = parseRecord(model);
  if (!parsed) return;
  try {
    store.setItem(DEFAULT_CHAT_MODEL_STORAGE_KEY, JSON.stringify(parsed));
    if (typeof window !== "undefined") {
      window.dispatchEvent(new Event("unsloth-default-chat-model-changed"));
    }
  } catch {
    // Best effort: private browsing and quota errors must not block model loading.
  }
}

export function clearDefaultChatModel(model?: DefaultChatModel): void {
  const store = storage();
  if (!store) return;
  if (model && !sameDefaultChatModel(readDefaultChatModel(), model)) return;
  try {
    store.removeItem(DEFAULT_CHAT_MODEL_STORAGE_KEY);
    if (typeof window !== "undefined") {
      window.dispatchEvent(new Event("unsloth-default-chat-model-changed"));
    }
  } catch {
    // Best effort only.
  }
}

export function isDefaultChatModel(model: DefaultChatModel): boolean {
  return sameDefaultChatModel(readDefaultChatModel(), model);
}

export function readDefaultChatModelLoadOnStartup(): boolean {
  const store = storage();
  if (!store) return false;
  try {
    return store.getItem(DEFAULT_CHAT_MODEL_STARTUP_KEY) === "1";
  } catch {
    return false;
  }
}

export function setDefaultChatModelLoadOnStartup(enabled: boolean): void {
  const store = storage();
  if (!store) return;
  try {
    if (enabled) store.setItem(DEFAULT_CHAT_MODEL_STARTUP_KEY, "1");
    else store.removeItem(DEFAULT_CHAT_MODEL_STARTUP_KEY);
    if (typeof window !== "undefined") {
      window.dispatchEvent(new Event("unsloth-default-chat-model-changed"));
    }
  } catch {
    // Best effort only.
  }
}

export function isDefaultChatModelLoadOnStartup(): boolean {
  return readDefaultChatModelLoadOnStartup();
}

function sameDefaultChatModel(
  a: DefaultChatModel | null,
  b: DefaultChatModel,
): boolean {
  return Boolean(
    a && a.id === b.id && a.kind === b.kind && a.ggufVariant === b.ggufVariant,
  );
}
