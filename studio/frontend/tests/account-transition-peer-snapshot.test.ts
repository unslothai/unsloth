// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { stripTypeScriptTypes } from "node:module";
import test from "node:test";
import vm from "node:vm";
import {
  BROWSER_ACCOUNT_KEY,
  installAccountTransitionListener,
  transitionBrowserAccount,
} from "../src/lib/account-transition.ts";

// Reuse the checked-in DOM/VM harness; execute the shipped snapshot script unchanged.
const sibling = readFileSync(
  new URL("./reload-snapshot.test.ts", import.meta.url),
  "utf8",
);
const harness = sibling.slice(
  sibling.indexOf("type Listener ="),
  sibling.indexOf("function storedSnapshot("),
);
const createEnvironment = new Function(
  "vm",
  "script",
  stripTypeScriptTypes(harness) + "\nreturn createEnvironment;",
)(
  vm,
  readFileSync(
    new URL("../public/reload-snapshot.js", import.meta.url),
    "utf8",
  ),
);

function storage(values: Map<string, string>): Storage {
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

for (const supportsPageSwap of [true, false]) {
  test(`account-switch reload must not paint Alice's chat for Bob (${supportsPageSwap ? "pageswap" : "pagehide"})`, async () => {
    const shared = new Map([
      [BROWSER_ACCOUNT_KEY, "account:a1:alice"],
      ["unsloth_auth_token", "alice-token"],
    ]);
    const localStorage = storage(shared);
    const outgoing = createEnvironment({
      navigationType: "navigate",
      localStorage: shared,
      supportsPageSwap,
      rootHtml: "<main>Alice confidential acquisition discussion</main>",
      styleSheets: ["/assets/index-abc123.css"],
    });
    let listener: (event: Partial<StorageEvent>) => void;
    let incoming: ReturnType<typeof createEnvironment> | undefined;
    const peer = {
      localStorage,
      sessionStorage: storage(outgoing.storage),
      addEventListener: (_type: string, fn: typeof listener) => {
        listener = fn;
      },
      location: {
        reload() {
          assert.equal(shared.get("unsloth_auth_token"), "bob-token");
          if (supportsPageSwap)
            outgoing.dispatch("pageswap", {
              activation: { navigationType: "reload" },
            });
          else outgoing.dispatch("pagehide", { persisted: false });
          incoming = createEnvironment({
            navigationType: "reload",
            storage: outgoing.storage,
            localStorage: shared,
          });
          incoming.loadStyleSheets();
        },
      },
    } as unknown as Window;
    installAccountTransitionListener(peer);
    const login = {
      localStorage,
      sessionStorage: storage(new Map()),
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
      () => shared.set("unsloth_auth_token", "bob-token"),
      login,
    );
    listener!({
      key: BROWSER_ACCOUNT_KEY,
      oldValue: "account:a1:alice",
      newValue: shared.get(BROWSER_ACCOUNT_KEY),
      storageArea: localStorage,
    });
    assert.ok(incoming, "the peer reloaded with Bob's shared session");
    assert.doesNotMatch(
      incoming.shell?.html ?? "",
      /Alice confidential/,
      "Bob's reload paints Alice's private chat",
    );
  });
}
