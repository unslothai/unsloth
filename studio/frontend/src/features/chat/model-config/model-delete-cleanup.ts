// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { notifyInventoryChanged } from "@/stores/inventory-events";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import { ggufVariantsMatch, modelIdsMatch } from "./model-identity";
import {
  deletePerModelConfig,
  deletePerModelConfigsForModel,
} from "./per-model-config";
import { removeRecentModel } from "./recent-models";

type DeletedModelPayload = {
  id: string;
  eventId: string;
  ggufVariant?: string | null;
};

const CHANNEL_NAME = "unsloth.model-deleted";
const STORAGE_KEY = "unsloth.model-deleted";
const seenEvents = new Set<string>();
const seenEventOrder: string[] = [];

let syncReady = false;
let channel: BroadcastChannel | null = null;
let storageHandler: ((event: StorageEvent) => void) | null = null;
let syncUsers = 0;

function rememberEvent(id: string): boolean {
  if (seenEvents.has(id)) {
    return false;
  }
  seenEvents.add(id);
  seenEventOrder.push(id);
  while (seenEventOrder.length > 32) {
    const oldest = seenEventOrder.shift();
    if (oldest) {
      seenEvents.delete(oldest);
    }
  }
  return true;
}

function parsePayload(raw: unknown): DeletedModelPayload | null {
  if (!raw || typeof raw !== "object") {
    return null;
  }
  const payload = raw as Partial<DeletedModelPayload>;
  if (typeof payload.id !== "string" || typeof payload.eventId !== "string") {
    return null;
  }
  if (
    payload.ggufVariant !== undefined &&
    payload.ggufVariant !== null &&
    typeof payload.ggufVariant !== "string"
  ) {
    return null;
  }
  return {
    id: payload.id,
    eventId: payload.eventId,
    ...(payload.ggufVariant !== undefined
      ? { ggufVariant: payload.ggufVariant }
      : {}),
  };
}

function activeModelMatchesDeleted(payload: DeletedModelPayload): boolean {
  const state = useChatRuntimeStore.getState();
  const checkpoint = state.params.checkpoint;
  if (!checkpoint) {
    return false;
  }
  if (!modelIdsMatch(checkpoint, payload.id)) {
    return false;
  }
  if (payload.ggufVariant === undefined) {
    return true;
  }
  return ggufVariantsMatch(state.activeGgufVariant, payload.ggufVariant);
}

function applyDeletedModel(payload: DeletedModelPayload): void {
  if (payload.ggufVariant === undefined) {
    deletePerModelConfigsForModel(payload.id);
    removeRecentModel(payload.id, "all");
  } else {
    deletePerModelConfig(payload.id, payload.ggufVariant);
    removeRecentModel(payload.id, { variant: payload.ggufVariant });
  }
  if (activeModelMatchesDeleted(payload)) {
    useChatRuntimeStore.getState().clearCheckpoint();
  }
}

function handleRemotePayload(raw: unknown): void {
  const payload = parsePayload(raw);
  if (!(payload && rememberEvent(payload.eventId))) {
    return;
  }
  applyDeletedModel(payload);
}

function ensureDeleteSync(): void {
  if (syncReady || typeof window === "undefined") {
    return;
  }
  syncReady = true;
  if (typeof BroadcastChannel !== "undefined") {
    channel = new BroadcastChannel(CHANNEL_NAME);
    channel.onmessage = (event) => handleRemotePayload(event.data);
  }
  storageHandler = (event) => {
    if (event.key !== STORAGE_KEY || typeof event.newValue !== "string") {
      return;
    }
    try {
      handleRemotePayload(JSON.parse(event.newValue));
    } catch {
      return;
    }
  };
  window.addEventListener("storage", storageHandler);
}

function stopDeleteSync(): void {
  if (!syncReady) return;
  channel?.close();
  channel = null;
  if (storageHandler && typeof window !== "undefined") {
    window.removeEventListener("storage", storageHandler);
  }
  storageHandler = null;
  syncReady = false;
}

export function startModelDeleteSync(): () => void {
  if (typeof window === "undefined") {
    return () => {};
  }
  syncUsers += 1;
  ensureDeleteSync();
  let disposed = false;
  return () => {
    if (disposed) return;
    disposed = true;
    syncUsers = Math.max(0, syncUsers - 1);
    if (syncUsers === 0) stopDeleteSync();
  };
}

export function notifyModelDeleted(
  modelId: string,
  ggufVariant?: string | null,
  options: { notifyInventory?: boolean } = {},
): void {
  ensureDeleteSync();
  const payload: DeletedModelPayload = {
    id: modelId,
    eventId:
      typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
        ? crypto.randomUUID()
        : `${Date.now()}:${Math.random().toString(36).slice(2)}`,
    ...(ggufVariant !== undefined ? { ggufVariant } : {}),
  };
  rememberEvent(payload.eventId);
  applyDeletedModel(payload);
  if (options.notifyInventory !== false) {
    notifyInventoryChanged();
  }
  channel?.postMessage(payload);
  if (typeof window === "undefined") {
    return;
  }
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
  } catch {
    return;
  }
}
