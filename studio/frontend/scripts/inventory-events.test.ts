// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

class MemoryStorage implements Storage {
  private readonly map = new Map<string, string>();

  get length(): number {
    return this.map.size;
  }

  clear(): void {
    this.map.clear();
  }

  getItem(key: string): string | null {
    return this.map.get(key) ?? null;
  }

  key(index: number): string | null {
    return Array.from(this.map.keys())[index] ?? null;
  }

  removeItem(key: string): void {
    this.map.delete(key);
  }

  setItem(key: string, value: string): void {
    this.map.set(key, value);
  }
}

test("inventory subscription applies the current cross-tab bump snapshot", async () => {
  const storage = new MemoryStorage();
  storage.setItem("unsloth.inventory.bump", "remote:1");
  const windowShim = {
    localStorage: storage,
    addEventListener() {},
    removeEventListener() {},
  };
  Object.defineProperty(globalThis, "window", {
    value: windowShim,
    configurable: true,
  });
  Object.defineProperty(globalThis, "BroadcastChannel", {
    value: undefined,
    configurable: true,
  });

  const moduleUrl = new URL(
    "../src/stores/inventory-events.ts",
    import.meta.url,
  );
  moduleUrl.search = `?snapshot=${Date.now()}`;
  const { getInventoryVersion, subscribeInventory } = await import(
    moduleUrl.href
  );
  const before = getInventoryVersion();
  let calls = 0;
  const unsubscribe = subscribeInventory(() => {
    calls += 1;
  });

  assert.equal(getInventoryVersion(), before + 1);
  assert.equal(calls, 1);
  unsubscribe();
});

test("inventory subscription tears down cross-tab listeners", async () => {
  const storage = new MemoryStorage();
  let storageAdds = 0;
  let storageRemoves = 0;
  const windowShim = {
    localStorage: storage,
    addEventListener(type: string) {
      if (type === "storage") storageAdds += 1;
    },
    removeEventListener(type: string) {
      if (type === "storage") storageRemoves += 1;
    },
  };
  const channels: Array<{ name: string; closed: boolean }> = [];
  class MockBroadcastChannel {
    readonly name: string;
    closed = false;
    onmessage: ((event: MessageEvent) => void) | null = null;

    constructor(name: string) {
      this.name = name;
      channels.push(this);
    }

    postMessage(): void {}

    close(): void {
      this.closed = true;
    }
  }
  Object.defineProperty(globalThis, "window", {
    value: windowShim,
    configurable: true,
  });
  Object.defineProperty(globalThis, "BroadcastChannel", {
    value: MockBroadcastChannel,
    configurable: true,
  });

  const moduleUrl = new URL(
    "../src/stores/inventory-events.ts",
    import.meta.url,
  );
  moduleUrl.search = `?teardown=${Date.now()}`;
  const { subscribeInventory } = await import(moduleUrl.href);

  const unsubscribe = subscribeInventory(() => {});
  assert.equal(storageAdds, 1);
  assert.equal(channels.length, 1);
  assert.equal(channels[0].closed, false);

  unsubscribe();
  assert.equal(storageRemoves, 1);
  assert.equal(channels[0].closed, true);
});

test("inventory bump before subscription writes snapshot and delays outbound close", async () => {
  const storage = new MemoryStorage();
  const windowShim = {
    localStorage: storage,
    addEventListener() {},
    removeEventListener() {},
  };
  const channels: Array<{ closed: boolean; messages: string[] }> = [];
  class MockBroadcastChannel {
    closed = false;
    messages: string[] = [];

    constructor() {
      channels.push(this);
    }

    postMessage(message: string): void {
      this.messages.push(message);
    }

    close(): void {
      this.closed = true;
    }
  }
  const timers = new Map<number, () => void>();
  let nextTimerId = 1;
  const originalSetTimeout = globalThis.setTimeout;
  const originalClearTimeout = globalThis.clearTimeout;

  Object.defineProperty(globalThis, "window", {
    value: windowShim,
    configurable: true,
  });
  Object.defineProperty(globalThis, "BroadcastChannel", {
    value: MockBroadcastChannel,
    configurable: true,
  });
  Object.defineProperty(globalThis, "setTimeout", {
    value: (callback: () => void) => {
      const id = nextTimerId++;
      timers.set(id, callback);
      return id;
    },
    configurable: true,
  });
  Object.defineProperty(globalThis, "clearTimeout", {
    value: (id: number) => {
      timers.delete(id);
    },
    configurable: true,
  });

  try {
    const moduleUrl = new URL(
      "../src/stores/inventory-events.ts",
      import.meta.url,
    );
    moduleUrl.search = `?preSub=${Date.now()}`;
    const { getInventoryVersion, notifyInventoryChanged } = await import(
      moduleUrl.href
    );

    const before = getInventoryVersion();
    notifyInventoryChanged();
    assert.equal(getInventoryVersion(), before + 1);
    const stored = storage.getItem("unsloth.inventory.bump");
    assert.equal(typeof stored, "string");
    assert.equal(channels.length, 1);
    assert.equal(channels[0].closed, false);
    assert.deepEqual(channels[0].messages, [stored]);

    notifyInventoryChanged();
    assert.equal(channels.length, 1);
    assert.equal(channels[0].closed, false);
    assert.equal(channels[0].messages.length, 2);

    const pendingCallbacks = Array.from(timers.values());
    timers.clear();
    for (const callback of pendingCallbacks) {
      callback();
    }
    assert.equal(channels[0].closed, true);
  } finally {
    Object.defineProperty(globalThis, "setTimeout", {
      value: originalSetTimeout,
      configurable: true,
    });
    Object.defineProperty(globalThis, "clearTimeout", {
      value: originalClearTimeout,
      configurable: true,
    });
  }
});
