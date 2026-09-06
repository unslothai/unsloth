// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  ACCOUNT_CHROME_KEYS,
  ACCOUNT_DATABASES,
  BROWSER_ACCOUNT_KEY,
  installAccountTransitionListener,
  resetFullAccessForMultiUser,
  transitionBrowserAccount,
} from "../src/lib/account-transition.ts";

function browserWith(
  values: Record<string, string> = {},
  databaseResult = "success",
) {
  const data = new Map(Object.entries(values));
  const removed: string[] = [];
  const deleted: string[] = [];
  const replaced: string[] = [];
  let reloads = 0;
  const listeners: ((event: Partial<StorageEvent>) => void)[] = [];
  const storage = {
    get length() {
      return data.size;
    },
    key: (index: number) => [...data.keys()][index] ?? null,
    getItem: (key: string) => data.get(key) ?? null,
    setItem: (key: string, value: string) => {
      data.set(key, value);
    },
    removeItem: (key: string) => {
      removed.push(key);
      data.delete(key);
    },
  } as Storage;
  const browser = {
    localStorage: storage,
    indexedDB: {
      deleteDatabase: (name: string) => {
        deleted.push(name);
        const request = {} as IDBOpenDBRequest;
        queueMicrotask(() => {
          const callback =
            databaseResult === "blocked"
              ? request.onblocked
              : databaseResult === "error"
                ? request.onerror
                : request.onsuccess;
          callback?.call(request, {} as IDBVersionChangeEvent);
        });
        return request;
      },
    },
    location: {
      replace: (route: string) => {
        replaced.push(route);
      },
      reload: () => {
        reloads++;
      },
    },
    addEventListener: (
      _type: string,
      listener: (event: Partial<StorageEvent>) => void,
    ) => {
      listeners.push(listener);
    },
  } as unknown as Window;
  return {
    browser,
    data,
    removed,
    deleted,
    replaced,
    listeners,
    reloads: () => reloads,
  };
}

for (const marker of [undefined, "unsloth"]) {
  test(`single-user login never purges, marker ${marker ?? "absent"}`, async () => {
    const b = browserWith({
      unsloth_chat_permission_mode: "full",
      "chat-draft:1": "keep",
      ...(marker ? { [BROWSER_ACCOUNT_KEY]: marker } : {}),
    });
    let committed = 0;
    assert.equal(
      await transitionBrowserAccount(
        " UNSLOTH ",
        "/chat",
        () => {
          committed++;
        },
        b.browser,
      ),
      false,
    );
    assert.equal(committed, 1);
    assert.deepEqual(b.removed, []);
    assert.deepEqual(b.deleted, []);
    assert.deepEqual(b.replaced, []);
    assert.equal(b.data.get("unsloth_chat_permission_mode"), "full");
  });
}

test("switch removes every content prefix and preserves only listed chrome and unrelated keys", async () => {
  const chrome = Object.fromEntries(
    [...ACCOUNT_CHROME_KEYS].map((key) => [key, "chrome"]),
  );
  const content = [
    "unsloth_auth_token",
    "unsloth_hf_token",
    "unsloth_new_feature",
    "unsloth_chat_permission_mode",
    "unsloth-profile",
    "chat-draft:1",
    "chat-draft-pastes:2",
  ];
  const b = browserWith({
    ...chrome,
    ...Object.fromEntries(content.map((key) => [key, "private"])),
    unrelated: "keep",
    "unsloth_web_update_dismissed:pip:1": "keep",
    [BROWSER_ACCOUNT_KEY]: "unsloth",
  });
  const changed = await transitionBrowserAccount(
    "Alice",
    "/change-password",
    () => {
      assert.deepEqual(b.deleted, [...ACCOUNT_DATABASES]);
      assert.equal(b.data.get(BROWSER_ACCOUNT_KEY), "unsloth");
      b.data.set("unsloth_auth_token", "alice-token");
    },
    b.browser,
  );
  assert.equal(changed, true);
  assert.deepEqual(b.removed.sort(), content.sort());
  for (const key of ACCOUNT_CHROME_KEYS)
    assert.equal(b.data.get(key), "chrome");
  assert.equal(b.data.get("unrelated"), "keep");
  assert.equal(b.data.get("unsloth_web_update_dismissed:pip:1"), "keep");
  assert.equal(b.data.get("unsloth_auth_token"), "alice-token");
  assert.equal(b.data.get(BROWSER_ACCOUNT_KEY), "alice");
  assert.deepEqual(b.replaced, ["/change-password"]);
});

test("a first managed login clears legacy owner data even without a marker", async () => {
  const b = browserWith({ "unsloth-old-content": "owner" });
  assert.equal(
    await transitionBrowserAccount("alice", "/chat", () => {}, b.browser),
    true,
  );
  assert.deepEqual(b.removed, ["unsloth-old-content"]);
});

test("same managed account keeps content and avoids IndexedDB work", async () => {
  const b = browserWith({
    [BROWSER_ACCOUNT_KEY]: "alice",
    "chat-draft:1": "alice",
  });
  await transitionBrowserAccount("ALICE", "/chat", () => {}, b.browser);
  assert.deepEqual(b.removed, []);
  assert.deepEqual(b.deleted, []);
  assert.deepEqual(b.replaced, []);
});

test("returning to the owner clears the previous managed account", async () => {
  const b = browserWith({
    [BROWSER_ACCOUNT_KEY]: "alice",
    "unsloth-private": "alice",
  });
  assert.equal(
    await transitionBrowserAccount("unsloth", "/chat", () => {}, b.browser),
    true,
  );
  assert.equal(b.data.has("unsloth-private"), false);
});

for (const failure of ["blocked", "error"]) {
  test(`IndexedDB ${failure} prevents new session publication and navigation`, async () => {
    const b = browserWith({ [BROWSER_ACCOUNT_KEY]: "unsloth" }, failure);
    let committed = false;
    await assert.rejects(
      transitionBrowserAccount(
        "alice",
        "/chat",
        () => {
          committed = true;
        },
        b.browser,
      ),
    );
    assert.equal(committed, false);
    assert.equal(b.data.get(BROWSER_ACCOUNT_KEY), "unsloth");
    assert.deepEqual(b.replaced, []);
  });
}

test("cross-tab switches reload once, ignoring initial owner markers, removals and unrelated storage", () => {
  const b = browserWith();
  installAccountTransitionListener(b.browser);
  installAccountTransitionListener(b.browser);
  assert.equal(b.listeners.length, 1);
  const send = b.listeners[0];
  send({ key: BROWSER_ACCOUNT_KEY, oldValue: null, newValue: "unsloth" });
  send({ key: "unrelated", oldValue: "a", newValue: "b" });
  send({ key: BROWSER_ACCOUNT_KEY, oldValue: "alice", newValue: null });
  send({ key: BROWSER_ACCOUNT_KEY, oldValue: "alice", newValue: "ALICE" });
  send({
    key: BROWSER_ACCOUNT_KEY,
    oldValue: "unsloth",
    newValue: "alice",
    storageArea: {} as Storage,
  });
  assert.equal(b.reloads(), 0);
  send({
    key: BROWSER_ACCOUNT_KEY,
    oldValue: "unsloth",
    newValue: "alice",
    storageArea: b.browser.localStorage,
  });
  send({ key: BROWSER_ACCOUNT_KEY, oldValue: "alice", newValue: "bob" });
  assert.equal(b.reloads(), 1);
});

test("multi-user policy resets full while preserving other permission modes", () => {
  const b = browserWith({ unsloth_chat_permission_mode: "full" });
  resetFullAccessForMultiUser(b.browser.localStorage);
  assert.equal(b.data.get("unsloth_chat_permission_mode"), "auto");
  for (const mode of ["ask", "auto", "off"]) {
    b.data.set("unsloth_chat_permission_mode", mode);
    resetFullAccessForMultiUser(b.browser.localStorage);
    assert.equal(b.data.get("unsloth_chat_permission_mode"), mode);
  }
});
