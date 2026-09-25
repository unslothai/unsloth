// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  BROWSER_ACCOUNT_KEY,
  transitionBrowserAccount,
} from "../src/lib/account-transition.ts";
import {
  dismissExample,
  isExampleDismissed,
  readLastPrompt,
  saveLastPrompt,
} from "../src/lib/last-prompt.ts";
import { WORKFLOW_EXAMPLE_PROMPTS, WORKFLOW_TABS } from "../src/features/images/workflows.ts";

function withStorage(storage: unknown, run: () => void) {
  const prev = Object.getOwnPropertyDescriptor(globalThis, "localStorage");
  Object.defineProperty(globalThis, "localStorage", { value: storage, configurable: true });
  try {
    run();
  } finally {
    if (prev) Object.defineProperty(globalThis, "localStorage", prev);
    else delete (globalThis as { localStorage?: unknown }).localStorage;
  }
}

function memoryStorage() {
  const store = new Map<string, string>();
  return {
    getItem: (k: string) => store.get(k) ?? null,
    setItem: (k: string, v: string) => void store.set(k, v),
  };
}

test("the last prompt survives a reload, per page and workflow", () => {
  withStorage(memoryStorage(), () => {
    assert.equal(readLastPrompt("images:edit"), "");
    saveLastPrompt("images:edit", "make it blue");
    assert.equal(readLastPrompt("images:edit"), "make it blue");
    assert.equal(readLastPrompt("images:create"), "");
  });
});

test("an example hint dismissed once stays dismissed for that box", () => {
  withStorage(memoryStorage(), () => {
    assert.equal(isExampleDismissed("images:edit"), false);
    dismissExample("images:edit");
    assert.equal(isExampleDismissed("images:edit"), true);
    assert.equal(isExampleDismissed("images:create"), false);
  });
});

test("unavailable storage reads as nothing saved and the hint shown", () => {
  const broken = {
    getItem: () => {
      throw new Error("denied");
    },
    setItem: () => {
      throw new Error("denied");
    },
  };
  withStorage(broken, () => {
    saveLastPrompt("video", "x");
    dismissExample("video");
    assert.equal(readLastPrompt("video"), "");
    assert.equal(isExampleDismissed("video"), false);
  });
});

test("every image workflow has its own short example prompt", () => {
  const examples = WORKFLOW_TABS.map(({ id }) => WORKFLOW_EXAMPLE_PROMPTS[id]);
  assert.equal(new Set(examples).size, WORKFLOW_TABS.length);
  for (const example of examples) {
    assert.ok(example.length > 0 && example.length < 200);
    assert.ok(!example.includes("\u2014"));
  }
});

function accountBrowser(account: string) {
  const data = new Map([[BROWSER_ACCOUNT_KEY, account]]);
  const localStorage = {
    get length() {
      return data.size;
    },
    key: (index: number) => [...data.keys()][index] ?? null,
    getItem: (key: string) => data.get(key) ?? null,
    setItem: (key: string, value: string) => void data.set(key, value),
    removeItem: (key: string) => void data.delete(key),
  } as unknown as Storage;
  const browser = {
    localStorage,
    sessionStorage: { length: 0, key: () => null, removeItem: () => {} },
    indexedDB: {
      deleteDatabase: () => {
        const request = {} as IDBOpenDBRequest;
        queueMicrotask(() => request.onsuccess?.call(request, {} as IDBVersionChangeEvent));
        return request;
      },
    },
    location: { replace: () => {} },
  } as unknown as Window;
  return { browser, localStorage };
}

test("another account signing in does not see the previous account's prompts", async () => {
  const { browser, localStorage } = accountBrowser("alice");
  withStorage(localStorage, () => {
    saveLastPrompt("images:create", "alice's prompt");
    dismissExample("images:create");
  });
  await transitionBrowserAccount("bob", "/images", () => {}, browser);
  withStorage(localStorage, () => {
    assert.equal(readLastPrompt("images:create"), "");
    assert.equal(isExampleDismissed("images:create"), false);
  });
});

test("the same account signing in again keeps its prompts", async () => {
  const { browser, localStorage } = accountBrowser("alice");
  withStorage(localStorage, () => saveLastPrompt("video", "alice's clip"));
  await transitionBrowserAccount("alice", "/video", () => {}, browser);
  withStorage(localStorage, () => {
    assert.equal(readLastPrompt("video"), "alice's clip");
  });
});

