// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  getTrainingCompareHandoff,
  setTrainingCompareHandoff,
} from "../src/features/chat/lib/training-compare-handoff.ts";
import {
  BROWSER_ACCOUNT_KEY,
  installAccountTransitionListener,
  transitionBrowserAccount,
} from "../src/lib/account-transition.ts";

function storage(): Storage {
  const values = new Map<string, string>();
  return {
    get length() {
      return values.size;
    },
    key: (i: number) => [...values.keys()][i] ?? null,
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => {
      values.set(key, value);
    },
    removeItem: (key: string) => {
      values.delete(key);
    },
    clear: () => values.clear(),
  };
}

test("peer-tab account switch clears Alice's training handoff before Bob reloads", async () => {
  const localStorage = storage();
  localStorage.setItem(BROWSER_ACCOUNT_KEY, "account:a1:alice");
  localStorage.setItem("unsloth_auth_token", "alice-token");
  const peerSession = storage();
  peerSession.setItem("unsloth_server_user_stopped", "1");
  let listener: (event: Partial<StorageEvent>) => void;
  let handoffAtReload: unknown;
  let reloads = 0;
  const peer = {
    localStorage,
    sessionStorage: peerSession,
    addEventListener: (_type: string, fn: typeof listener) => {
      listener = fn;
    },
    location: {
      reload() {
        reloads++;
        assert.equal(localStorage.getItem("unsloth_auth_token"), "bob-token");
        // A real reload preserves sessionStorage. Read with the actual Chat consumer.
        handoffAtReload = getTrainingCompareHandoff();
      },
    },
  } as unknown as Window;
  const priorWindow = Object.getOwnPropertyDescriptor(globalThis, "window");
  Object.defineProperty(globalThis, "window", {
    value: peer,
    configurable: true,
  });
  try {
    setTrainingCompareHandoff("alice-private/confidential-finetune");
    installAccountTransitionListener(peer);
    const loginTab = {
      localStorage,
      sessionStorage: storage(),
      indexedDB: {
        deleteDatabase() {
          const request = {} as IDBOpenDBRequest;
          queueMicrotask(() => request.onsuccess?.call(request, {} as Event));
          return request;
        },
      },
      location: { replace() {} },
    } as unknown as Window;
    await transitionBrowserAccount(
      { username: "bob", accountId: "b1" },
      "/chat",
      () => localStorage.setItem("unsloth_auth_token", "bob-token"),
      loginTab,
    );
    listener!({
      key: BROWSER_ACCOUNT_KEY,
      oldValue: "account:a1:alice",
      newValue: localStorage.getItem(BROWSER_ACCOUNT_KEY),
      storageArea: localStorage,
    });
    assert.equal(reloads, 1);
    assert.equal(peerSession.getItem("unsloth_server_user_stopped"), "1");
    assert.equal(
      handoffAtReload,
      null,
      "Bob inherits Alice's pending training comparison",
    );
  } finally {
    if (priorWindow) Object.defineProperty(globalThis, "window", priorWindow);
    else delete (globalThis as { window?: Window }).window;
  }
});
