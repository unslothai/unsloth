// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useSyncExternalStore } from "react";

type Listener = () => void;

const CHANNEL_NAME = "unsloth.inventory";
const STORAGE_KEY = "unsloth.inventory.bump";
const OUTBOUND_CHANNEL_CLOSE_DELAY_MS = 250;

let _version = 0;
const listeners = new Set<Listener>();
const seenBumps = new Set<string>();
const seenBumpOrder: string[] = [];
let crossTabReady = false;
let channel: BroadcastChannel | null = null;
let outboundChannel: BroadcastChannel | null = null;
let outboundCloseTimer: ReturnType<typeof setTimeout> | null = null;
let storageHandler: ((event: StorageEvent) => void) | null = null;

function emitInventoryChange(): void {
  for (const fn of [...listeners]) fn();
}

function rememberBump(id: string): boolean {
  if (seenBumps.has(id)) return false;
  seenBumps.add(id);
  seenBumpOrder.push(id);
  while (seenBumpOrder.length > 32) {
    const oldest = seenBumpOrder.shift();
    if (oldest) seenBumps.delete(oldest);
  }
  return true;
}

function applyInventoryBump(): void {
  _version += 1;
  emitInventoryChange();
}

function ensureCrossTabSync(): void {
  if (crossTabReady || typeof window === "undefined") return;
  crossTabReady = true;
  if (typeof BroadcastChannel !== "undefined") {
    channel = new BroadcastChannel(CHANNEL_NAME);
    channel.onmessage = (event) => {
      if (typeof event.data !== "string") return;
      if (!rememberBump(event.data)) return;
      applyInventoryBump();
    };
  }
  storageHandler = (event) => {
    if (event.key !== STORAGE_KEY || typeof event.newValue !== "string") return;
    if (!rememberBump(event.newValue)) return;
    applyInventoryBump();
  };
  window.addEventListener("storage", storageHandler);
  try {
    const current = window.localStorage.getItem(STORAGE_KEY);
    if (typeof current === "string") rememberBump(current);
  } catch {
    void 0;
  }
}

function stopCrossTabSync(): void {
  if (!crossTabReady) return;
  channel?.close();
  channel = null;
  if (storageHandler && typeof window !== "undefined") {
    window.removeEventListener("storage", storageHandler);
  }
  storageHandler = null;
  crossTabReady = false;
}

function closeOutboundChannel(): void {
  outboundCloseTimer = null;
  outboundChannel?.close();
  outboundChannel = null;
}

function getOutboundChannel(): BroadcastChannel | null {
  if (typeof BroadcastChannel === "undefined") {
    return null;
  }
  if (!outboundChannel) {
    outboundChannel = new BroadcastChannel(CHANNEL_NAME);
  }
  if (outboundCloseTimer !== null) {
    clearTimeout(outboundCloseTimer);
  }
  outboundCloseTimer = setTimeout(
    closeOutboundChannel,
    OUTBOUND_CHANNEL_CLOSE_DELAY_MS,
  );
  return outboundChannel;
}

function postInventoryBump(id: string): void {
  const target = channel ?? getOutboundChannel();
  target?.postMessage(id);
  try {
    window.localStorage.setItem(STORAGE_KEY, id);
  } catch {
    void 0;
  }
}

export function getInventoryVersion(): number {
  return _version;
}

export function bumpInventoryVersion(): void {
  const id =
    typeof window === "undefined"
      ? null
      : `${Date.now()}:${Math.random().toString(36).slice(2)}`;
  if (id) rememberBump(id);
  applyInventoryBump();
  if (!id) return;
  postInventoryBump(id);
}

export function subscribeInventory(listener: Listener): () => void {
  listeners.add(listener);
  ensureCrossTabSync();
  return () => {
    listeners.delete(listener);
    if (listeners.size === 0) {
      stopCrossTabSync();
    }
  };
}

export function useInventoryVersion(): number {
  return useSyncExternalStore(
    subscribeInventory,
    getInventoryVersion,
    getInventoryVersion,
  );
}
