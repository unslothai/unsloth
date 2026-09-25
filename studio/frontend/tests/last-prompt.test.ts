// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  BROWSER_ACCOUNT_KEY,
  transitionBrowserAccount,
} from "../src/lib/account-transition.ts";
import { readLastPrompt, saveLastPrompt } from "../src/lib/last-prompt.ts";
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

test("the last prompt survives a reload and falls back to the example", () => {
  const store = new Map<string, string>();
  const storage = {
    getItem: (k: string) => store.get(k) ?? null,
    setItem: (k: string, v: string) => void store.set(k, v),
  };
  withStorage(storage, () => {
    assert.equal(readLastPrompt("images:edit", "example"), "example");
    saveLastPrompt("images:edit", "make it blue");
    assert.equal(readLastPrompt("images:edit", "example"), "make it blue");
    // Keys are separate per page and workflow.
    assert.equal(readLastPrompt("images:create", "example"), "example");
  });
});

test("unavailable storage still yields the example", () => {
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
    assert.equal(readLastPrompt("video", "example"), "example");
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
  withStorage(localStorage, () => saveLastPrompt("images:create", "alice's prompt"));
  await transitionBrowserAccount("bob", "/images", () => {}, browser);
  withStorage(localStorage, () => {
    assert.equal(readLastPrompt("images:create", "example"), "example");
  });
});

test("the same account signing in again keeps its prompts", async () => {
  const { browser, localStorage } = accountBrowser("alice");
  withStorage(localStorage, () => saveLastPrompt("video", "alice's clip"));
  await transitionBrowserAccount("alice", "/video", () => {}, browser);
  withStorage(localStorage, () => {
    assert.equal(readLastPrompt("video", "example"), "alice's clip");
  });
});

