// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The model picker's pins, mirrored to this account's studio.db. Switching accounts clears every
// "unsloth*" key in the browser, so without a server copy a switch lost every pin. localStorage stays
// the store the picker reads; this only restores it when it is empty and keeps the server current.

export type PinList = "pinned" | "connected";

const URL = "/api/settings/pinned-models";
const STORAGE_KEYS: Record<PinList, string> = {
  pinned: "unsloth_pinned_models",
  connected: "unsloth_pinned_connected_models",
};

const restorers: Partial<Record<PinList, () => void>> = {};
// Writes wait for hydration: a fresh browser's empty list must not land over the account's pins.
let hydrated = false;
let pending: Partial<Record<PinList, string[]>> = {};
let queue: Promise<void> = Promise.resolve();

async function request(init?: RequestInit): Promise<Response> {
  // Imported on first use, so the stores stay importable where there is no auth module to load.
  const { authFetch } = await import("@/features/auth");
  return authFetch(URL, init);
}

function send(body: Partial<Record<PinList, string[]>>): void {
  queue = queue
    .then(async () => {
      const res = await request({
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) console.warn(`Saving pinned models failed (${res.status})`);
    })
    .catch((error: unknown) =>
      console.warn("Saving pinned models failed", error),
    );
}

/** Registered by each pin store: re-read its list after hydration restored it. */
export function onPinsRestored(list: PinList, restore: () => void): void {
  restorers[list] = restore;
}

export function mirrorPins(list: PinList, ids: readonly string[]): void {
  if (!hydrated) {
    pending[list] = [...ids];
    return;
  }
  send({ [list]: [...ids] });
}

function readLocal(key: string): string[] | null {
  try {
    const raw = localStorage.getItem(key);
    if (raw === null) return null;
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed)
      ? parsed.filter((v): v is string => typeof v === "string")
      : null;
  } catch {
    return null;
  }
}

/** Once per page load, after auth. A list this browser lacks comes back from the server; a list the
 *  server never stored is seeded from this browser. When both hold one, the browser's stands until its
 *  next edit, the same rule the stores already follow between tabs. */
export async function hydratePins(): Promise<void> {
  if (hydrated) return;
  let server: Partial<Record<PinList, string[] | null>>;
  try {
    const res = await request();
    // Not saved this session rather than saved over: the next load retries.
    if (!res.ok) return;
    server = await res.json();
  } catch {
    return;
  }
  const seed: Partial<Record<PinList, string[]>> = {};
  for (const list of Object.keys(STORAGE_KEYS) as PinList[]) {
    const key = STORAGE_KEYS[list];
    const local = readLocal(key);
    const remote = server[list];
    if (local === null && Array.isArray(remote) && !(list in pending)) {
      try {
        localStorage.setItem(key, JSON.stringify(remote));
      } catch {
        continue;
      }
      restorers[list]?.();
    } else if (local !== null && remote == null) {
      seed[list] = local;
    }
  }
  hydrated = true;
  const body = { ...seed, ...pending };
  pending = {};
  if (Object.keys(body).length > 0) send(body);
}

/** Test hook: forget hydration and queued writes. */
export function resetPinsMirrorForTests(): void {
  hydrated = false;
  pending = {};
  queue = Promise.resolve();
}
