// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  ACCOUNT_CHROME_KEYS,
  ACCOUNT_DATABASES,
  BROWSER_ACCOUNT_FENCE_KEY,
  BROWSER_ACCOUNT_KEY,
  accountDatabaseName,
  accountTransitionPending,
  browserAccountMarker,
  installAccountTransitionListener,
  normalizeAccountUsername,
  resetFullAccessForMultiUser,
  transitionBrowserAccount,
} from "../src/lib/account-transition.ts";

function browserWith(
  values: Record<string, string> = {},
  databaseResult = "success",
  sessionValues: Record<string, string> = {},
) {
  const data = new Map(Object.entries(values));
  const sessionData = new Map(Object.entries(sessionValues));
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
  const sessionStorage = {
    get length() {
      return sessionData.size;
    },
    key: (index: number) => [...sessionData.keys()][index] ?? null,
    getItem: (key: string) => sessionData.get(key) ?? null,
    setItem: (key: string, value: string) => {
      sessionData.set(key, value);
    },
    removeItem: (key: string) => {
      sessionData.delete(key);
    },
  } as Storage;
  const browser = {
    localStorage: storage,
    sessionStorage,
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
    sessionData,
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
  // The fence is written before the tokens and lifted after the marker.
  assert.deepEqual(b.removed.sort(), [...content, BROWSER_ACCOUNT_FENCE_KEY].sort());
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
  assert.deepEqual(b.removed, ["unsloth-old-content", BROWSER_ACCOUNT_FENCE_KEY]);
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

test("a username created again as a different account inherits nothing", async () => {
  const b = browserWith({});
  await transitionBrowserAccount(
    { username: "alice", accountId: "a1" },
    "/chat",
    () => {},
    b.browser,
  );
  b.data.set("chat-draft:1", "first alice");
  b.data.set("unsloth_hf_token", "first alice");
  const deletedOnFirstLogin = b.deleted.length;
  const changed = await transitionBrowserAccount(
    { username: "alice", accountId: "a2" },
    "/chat",
    () => {},
    b.browser,
  );
  assert.equal(changed, true);
  assert.equal(b.data.has("chat-draft:1"), false);
  assert.equal(b.data.has("unsloth_hf_token"), false);
  assert.deepEqual(b.deleted.slice(deletedOnFirstLogin), [
    ...ACCOUNT_DATABASES,
  ]);
  assert.equal(b.data.get(BROWSER_ACCOUNT_KEY), "account:a2:alice");
});

test("one account keeps its data across a rename", async () => {
  const b = browserWith({
    [BROWSER_ACCOUNT_KEY]: browserAccountMarker({
      username: "alice",
      accountId: "a1",
    }),
    "chat-draft:1": "alice",
  });
  assert.equal(
    await transitionBrowserAccount(
      { username: "Alice2", accountId: "a1" },
      "/chat",
      () => {},
      b.browser,
    ),
    false,
  );
  assert.deepEqual(b.removed, []);
  assert.deepEqual(b.deleted, []);
  assert.equal(b.data.get(BROWSER_ACCOUNT_KEY), "account:a1:alice2");
});

test("a marker written before account ids upgrades in place", async () => {
  const b = browserWith({
    [BROWSER_ACCOUNT_KEY]: "alice",
    "chat-draft:1": "alice",
  });
  assert.equal(
    await transitionBrowserAccount(
      { username: "alice", accountId: "a1" },
      "/chat",
      () => {},
      b.browser,
    ),
    false,
  );
  assert.deepEqual(b.removed, []);
  assert.equal(b.data.get(BROWSER_ACCOUNT_KEY), "account:a1:alice");
});

test("a server that sends no account id keeps the session on the username", async () => {
  for (const account of [
    "alice",
    { username: "alice", accountId: null },
  ] as const) {
    const b = browserWith({
      [BROWSER_ACCOUNT_KEY]: "account:a1:alice",
      "chat-draft:1": "alice",
    });
    assert.equal(
      await transitionBrowserAccount(account, "/chat", () => {}, b.browser),
      false,
    );
    assert.deepEqual(b.removed, []);
    assert.equal(b.data.get(BROWSER_ACCOUNT_KEY), "alice");
  }
});

for (const marker of [undefined, "unsloth"]) {
  test(`an owner-only install never purges when ids arrive, marker ${marker ?? "absent"}`, async () => {
    const b = browserWith({
      "chat-draft:1": "keep",
      ...(marker ? { [BROWSER_ACCOUNT_KEY]: marker } : {}),
    });
    assert.equal(
      await transitionBrowserAccount(
        { username: "unsloth", accountId: "owner" },
        "/chat",
        () => {},
        b.browser,
      ),
      false,
    );
    assert.deepEqual(b.removed, []);
    assert.deepEqual(b.deleted, []);
    assert.deepEqual(b.replaced, []);
    assert.equal(b.data.get(BROWSER_ACCOUNT_KEY), "account:owner:unsloth");
  });
}

test("a marker no build wrote can only compare as a different account", async () => {
  const b = browserWith({
    [BROWSER_ACCOUNT_KEY]: "account:",
    "unsloth-private": "x",
  });
  assert.equal(
    await transitionBrowserAccount(
      { username: "alice", accountId: "a1" },
      "/chat",
      () => {},
      b.browser,
    ),
    true,
  );
  assert.equal(b.data.has("unsloth-private"), false);
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

test("a switch fences peers before the new tokens and lifts the fence after the marker", async () => {
  const b = browserWith({ [BROWSER_ACCOUNT_KEY]: "unsloth", unsloth_auth_token: "alice-token" });
  let fenceAtCommit: string | null = null;
  await transitionBrowserAccount(
    { username: "bob", accountId: "b1" },
    "/chat",
    () => {
      fenceAtCommit = b.data.get(BROWSER_ACCOUNT_FENCE_KEY) ?? null;
      b.data.set("unsloth_auth_token", "bob-token");
    },
    b.browser,
  );
  assert.equal(fenceAtCommit, "account:b1:bob");
  assert.equal(b.data.has(BROWSER_ACCOUNT_FENCE_KEY), false);
  assert.equal(b.data.get(BROWSER_ACCOUNT_KEY), "account:b1:bob");
});

test("a peer holds requests from the fence until the marker reloads it", () => {
  const b = browserWith({ [BROWSER_ACCOUNT_KEY]: "unsloth" });
  installAccountTransitionListener(b.browser);
  const send = b.listeners[0];
  assert.equal(accountTransitionPending(), false);
  send({ key: BROWSER_ACCOUNT_FENCE_KEY, oldValue: null, newValue: "unsloth" });
  assert.equal(accountTransitionPending(), false);
  send({ key: BROWSER_ACCOUNT_FENCE_KEY, oldValue: null, newValue: "account:b1:bob" });
  assert.equal(accountTransitionPending(), true);
  assert.equal(b.reloads(), 0);
  send({ key: BROWSER_ACCOUNT_KEY, oldValue: "unsloth", newValue: "account:b1:bob" });
  assert.equal(b.reloads(), 1);
  send({ key: BROWSER_ACCOUNT_FENCE_KEY, oldValue: "account:b1:bob", newValue: null });
  assert.equal(b.reloads(), 1);
});

test("cross-tab reloads follow the account id, not the name", () => {
  const upgrade = browserWith();
  installAccountTransitionListener(upgrade.browser);
  upgrade.listeners[0]({
    key: BROWSER_ACCOUNT_KEY,
    oldValue: "alice",
    newValue: "account:a1:alice",
    storageArea: upgrade.browser.localStorage,
  });
  assert.equal(upgrade.reloads(), 0);
  const recreated = browserWith();
  installAccountTransitionListener(recreated.browser);
  recreated.listeners[0]({
    key: BROWSER_ACCOUNT_KEY,
    oldValue: "account:a1:alice",
    newValue: "account:a2:alice",
    storageArea: recreated.browser.localStorage,
  });
  assert.equal(recreated.reloads(), 1);
});

test("a marker carries the account id and the normalized name", () => {
  assert.equal(browserAccountMarker(" ALICE "), "alice");
  assert.equal(
    browserAccountMarker({ username: " ALICE ", accountId: "a1" }),
    "account:a1:alice",
  );
  assert.throws(() =>
    browserAccountMarker({ username: "  ", accountId: "a1" }),
  );
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

test("username normalization matches the ASCII names the backend can store", () => {
  for (const [input, expected] of [
    [" UNSLOTH ", "unsloth"],
    ["Alice_01", "alice_01"],
    ["BOB-2", "bob-2"],
    [" Straße ", "straße"],
  ])
    assert.equal(normalizeAccountUsername(input), expected);
});

test("a legacy marker still compares on the name when neither side has an id", async () => {
  const same = browserWith({
    [BROWSER_ACCOUNT_KEY]: "alice",
    "unsloth-private": "alice",
  });
  assert.equal(
    await transitionBrowserAccount(" ALICE ", "/chat", () => {}, same.browser),
    false,
  );
  assert.equal(same.data.get("unsloth-private"), "alice");
  const other = browserWith({
    [BROWSER_ACCOUNT_KEY]: "alice",
    "unsloth-private": "alice",
  });
  assert.equal(
    await transitionBrowserAccount("bob", "/chat", () => {}, other.browser),
    true,
  );
  assert.equal(other.data.has("unsloth-private"), false);
});

test("switching accounts clears the legacy browser-only chat store", async () => {
  const b = browserWith({ [BROWSER_ACCOUNT_KEY]: "unsloth" });
  await transitionBrowserAccount("alice", "/chat", () => {}, b.browser);
  assert.ok(b.deleted.includes("unsloth-chat"));
});

test("the owner's own first login keeps the legacy chat store to import", async () => {
  const b = browserWith({});
  await transitionBrowserAccount("unsloth", "/chat", () => {}, b.browser);
  assert.deepEqual(b.deleted, []);
});

test("switching accounts clears session content and keeps neutral session flags", async () => {
  const b = browserWith({ [BROWSER_ACCOUNT_KEY]: "alice" }, "success", {
    "chat:training-compare-handoff:v1": JSON.stringify({
      intent: "compare",
      baseModel: "alice-private/base",
      requestedAt: Date.now(),
    }),
    "unsloth.reload-snapshot.v1": "<div>alice</div>",
    "data-recipes:open-learning-recipes": "1",
    // USER_STOPPED_KEY: neutral, and clearing it would restart a server the user stopped.
    unsloth_server_user_stopped: "1",
  });
  assert.equal(
    await transitionBrowserAccount("bob", "/chat", () => {}, b.browser),
    true,
  );
  assert.equal(b.sessionData.has("chat:training-compare-handoff:v1"), false);
  assert.equal(b.sessionData.has("unsloth.reload-snapshot.v1"), false);
  assert.equal(b.sessionData.has("data-recipes:open-learning-recipes"), false);
  assert.equal(b.sessionData.get("unsloth_server_user_stopped"), "1");
});

test("the same account keeps a pending compare handoff", async () => {
  const b = browserWith({ [BROWSER_ACCOUNT_KEY]: "alice" }, "success", {
    "chat:training-compare-handoff:v1": "pending",
  });
  assert.equal(
    await transitionBrowserAccount("ALICE", "/chat", () => {}, b.browser),
    false,
  );
  assert.equal(
    b.sessionData.get("chat:training-compare-handoff:v1"),
    "pending",
  );
});

test("unreadable session storage never fails a sign-in", async () => {
  const b = browserWith({ [BROWSER_ACCOUNT_KEY]: "alice" });
  Object.defineProperty(b.browser, "sessionStorage", {
    get() {
      throw new Error("blocked");
    },
  });
  assert.equal(
    await transitionBrowserAccount("bob", "/chat", () => {}, b.browser),
    true,
  );
});

test("saved recipes live in a per-account store and are never purged", () => {
  assert.equal(
    ACCOUNT_DATABASES.includes("unsloth-data-recipes" as never),
    false,
  );
  const stored = (marker: string | null) => ({ getItem: () => marker });
  assert.equal(
    accountDatabaseName("unsloth-data-recipes", stored(null)),
    "unsloth-data-recipes",
  );
  assert.equal(
    accountDatabaseName("unsloth-data-recipes", stored("unsloth")),
    "unsloth-data-recipes",
  );
  assert.equal(
    accountDatabaseName(
      "unsloth-data-recipes",
      stored("account:owner:unsloth"),
    ),
    "unsloth-data-recipes",
  );
  assert.equal(
    accountDatabaseName("unsloth-data-recipes", stored("account:a1:alice")),
    "unsloth-data-recipes:a1",
  );
  assert.equal(
    accountDatabaseName("unsloth-data-recipes", stored("alice")),
    "unsloth-data-recipes:alice",
  );
});
