// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const OCR_MODEL_LOCK_KEY = "unsloth.chat.temporaryOcrModelLock";
const OCR_MODEL_LOCK_EVENT = "unsloth:temporary-ocr-model-lock";
const OCR_MODEL_LOCK_TTL_MS = 2 * 60 * 1000;
const OCR_MODEL_LOCK_HEARTBEAT_MS = 30 * 1000;
const OCR_MODEL_LOCK_POLL_MS = 250;

interface OcrModelLockState {
  active: boolean;
  ownerId: string;
  startedAt: number;
  expiresAt: number;
}

export interface TemporaryOcrModelLease {
  ownerId: string;
  isActive: () => boolean;
  assertActive: () => void;
  release: () => void;
}

function now(): number {
  return Date.now();
}

function dispatchLockEvent(): void {
  if (typeof window === "undefined") return;
  window.dispatchEvent(new Event(OCR_MODEL_LOCK_EVENT));
}

function readState(): OcrModelLockState | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(OCR_MODEL_LOCK_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<OcrModelLockState>;
    if (!parsed.active || typeof parsed.expiresAt !== "number") return null;
    if (parsed.expiresAt <= now()) {
      window.localStorage.removeItem(OCR_MODEL_LOCK_KEY);
      dispatchLockEvent();
      return null;
    }
    return {
      active: true,
      ownerId: parsed.ownerId || "legacy",
      startedAt:
        typeof parsed.startedAt === "number"
          ? parsed.startedAt
          : parsed.expiresAt - OCR_MODEL_LOCK_TTL_MS,
      expiresAt: parsed.expiresAt,
    };
  } catch {
    return null;
  }
}

function writeState(state: OcrModelLockState): void {
  window.localStorage.setItem(OCR_MODEL_LOCK_KEY, JSON.stringify(state));
  dispatchLockEvent();
}

function removeState(): void {
  window.localStorage.removeItem(OCR_MODEL_LOCK_KEY);
  dispatchLockEvent();
}

function makeOwnerId(): string {
  const randomId = globalThis.crypto?.randomUUID?.();
  if (randomId) return randomId;
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

function tryAcquire(ownerId: string): boolean {
  if (typeof window === "undefined") return true;
  try {
    const current = readState();
    if (current && current.ownerId !== ownerId) return false;
    const state: OcrModelLockState = {
      active: true,
      ownerId,
      startedAt: current?.startedAt ?? now(),
      expiresAt: now() + OCR_MODEL_LOCK_TTL_MS,
    };
    writeState(state);
    return readState()?.ownerId === ownerId;
  } catch {
    throw new Error("Temporary OCR model lock storage is unavailable.");
  }
}

function refresh(ownerId: string): boolean {
  if (typeof window === "undefined") return true;
  try {
    const current = readState();
    if (!current || current.ownerId !== ownerId) return false;
    writeState({
      ...current,
      expiresAt: now() + OCR_MODEL_LOCK_TTL_MS,
    });
    return true;
  } catch {
    return false;
  }
}

function release(ownerId: string): void {
  if (typeof window === "undefined") return;
  try {
    const current = readState();
    if (!current || current.ownerId === ownerId) {
      removeState();
    }
  } catch {
    // Best-effort cross-tab guard; the local tab still uses modelLoading.
  }
}

function isOwnerActive(ownerId: string): boolean {
  if (typeof window === "undefined") return true;
  try {
    return readState()?.ownerId === ownerId;
  } catch {
    return false;
  }
}

function assertOwnerActive(ownerId: string, lost: () => boolean): void {
  if (lost() || !isOwnerActive(ownerId)) {
    throw new Error(
      "Temporary OCR model lock was lost before extraction completed.",
    );
  }
}

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw new DOMException("Aborted", "AbortError");
  }
}

function delay(ms: number, signal?: AbortSignal): Promise<void> {
  throwIfAborted(signal);
  return new Promise((resolve, reject) => {
    const cleanup = (): void => {
      signal?.removeEventListener("abort", onAbort);
    };
    const timeout = window.setTimeout(() => {
      cleanup();
      resolve();
    }, ms);
    const onAbort = (): void => {
      window.clearTimeout(timeout);
      cleanup();
      reject(new DOMException("Aborted", "AbortError"));
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

export function isTemporaryOcrModelBusy(): boolean {
  return readState()?.active === true;
}

export function setTemporaryOcrModelBusy(active: boolean): void {
  if (typeof window === "undefined") return;
  try {
    if (active) {
      const state: OcrModelLockState = {
        active: true,
        ownerId: "legacy",
        startedAt: now(),
        expiresAt: now() + OCR_MODEL_LOCK_TTL_MS,
      };
      writeState(state);
    } else {
      removeState();
    }
  } catch {
    // Best-effort cross-tab guard; the local tab still uses modelLoading.
  }
}

export async function acquireTemporaryOcrModelLease(
  signal?: AbortSignal,
): Promise<TemporaryOcrModelLease> {
  if (typeof window === "undefined") {
    return {
      ownerId: "server",
      isActive: () => true,
      assertActive: () => {},
      release: () => {},
    };
  }
  const ownerId = makeOwnerId();
  while (!tryAcquire(ownerId)) {
    await delay(OCR_MODEL_LOCK_POLL_MS, signal);
  }
  let lost = false;
  const heartbeat = window.setInterval(() => {
    if (!refresh(ownerId)) {
      lost = true;
      window.clearInterval(heartbeat);
    }
  }, OCR_MODEL_LOCK_HEARTBEAT_MS);
  return {
    ownerId,
    isActive: () => !lost && isOwnerActive(ownerId),
    assertActive: () => assertOwnerActive(ownerId, () => lost),
    release: () => {
      window.clearInterval(heartbeat);
      release(ownerId);
    },
  };
}

export async function waitForTemporaryOcrModelIdle(
  signal?: AbortSignal,
): Promise<void> {
  if (typeof window === "undefined") return;
  while (isTemporaryOcrModelBusy()) {
    await delay(OCR_MODEL_LOCK_POLL_MS, signal);
  }
}

export function subscribeTemporaryOcrModelBusy(
  onChange: () => void,
): () => void {
  if (typeof window === "undefined") return () => {};
  const onStorage = (event: StorageEvent): void => {
    if (event.key === OCR_MODEL_LOCK_KEY) onChange();
  };
  window.addEventListener("storage", onStorage);
  window.addEventListener(OCR_MODEL_LOCK_EVENT, onChange);
  return () => {
    window.removeEventListener("storage", onStorage);
    window.removeEventListener(OCR_MODEL_LOCK_EVENT, onChange);
  };
}
