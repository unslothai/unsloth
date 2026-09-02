// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The browser primitives the account transition is built on, exercised on every
 * engine the matrix runs.
 *
 * account-transition.ts is covered by unit tests against a fake storage, which is
 * where its rules belong. What a fake cannot answer is whether the three platform
 * behaviours it depends on are the same in Firefox, in WebKit and in a branded
 * Chrome as they are in the fake:
 *
 *   1. localStorage.key(i) enumerating every key, which is how the inverted purge
 *      finds a key nobody listed.
 *   2. a same-document removeItem firing NO storage event, and a cross-document
 *      write firing one carrying the NEW value. Both halves matter: the first is
 *      why an event exists at all, the second is why the cross-tab guard compares
 *      against a captured value rather than reading the key back.
 *   3. indexedDB.deleteDatabase making progress with a live connection open.
 *
 * Deliberately about the platform rather than the app, so it needs no session and
 * cannot go stale when the UI moves.
 */

import { expect, test } from "@playwright/test";

const baseURL = process.env.STUDIO_E2E_URL ?? "http://127.0.0.1:8767";
const LAST_ACCOUNT_KEY = "unsloth.browser-account.v1";

test.beforeEach(async ({ page }) => {
  await page.goto(baseURL);
  await page.evaluate(() => localStorage.clear());
});

test("every stored key is enumerable, including one nobody listed", async ({ page }) => {
  const found = await page.evaluate(() => {
    localStorage.setItem("unsloth_hf_token", "hf_secret");
    localStorage.setItem("unsloth.some.future.store.v3", "written by a later release");
    localStorage.setItem("chat-draft:42", "half a message");
    localStorage.setItem("third_party_thing", "not ours");
    const keys: string[] = [];
    for (let index = 0; index < localStorage.length; index += 1) {
      const key = localStorage.key(index);
      if (key !== null) keys.push(key);
    }
    return keys.sort();
  });
  expect(found).toEqual([
    "chat-draft:42",
    "third_party_thing",
    "unsloth.some.future.store.v3",
    "unsloth_hf_token",
  ]);
});

test("a same-document removeItem fires no storage event", async ({ page }) => {
  const events = await page.evaluate(async () => {
    const seen: string[] = [];
    window.addEventListener("storage", (event) => seen.push(String(event.key)));
    localStorage.setItem("unsloth_hf_token", "hf_secret");
    localStorage.removeItem("unsloth_hf_token");
    await new Promise((resolve) => setTimeout(resolve, 250));
    return seen;
  });
  expect(events, "an in-page purge must not be observable as a storage event").toEqual([]);
});

test("a write in another tab is delivered with the new value already readable", async ({
  context,
  page,
}) => {
  const other = await context.newPage();
  await other.goto(baseURL);

  const settled = page.evaluate(
    ({ key }) =>
      new Promise<{ newValue: string | null; readBack: string | null }>((resolve) => {
        window.addEventListener("storage", (event) => {
          if (event.key !== key) return;
          resolve({ newValue: event.newValue, readBack: localStorage.getItem(key) });
        });
        setTimeout(() => resolve({ newValue: "TIMEOUT", readBack: null }), 10_000);
      }),
    { key: LAST_ACCOUNT_KEY },
  );

  await other.evaluate(
    ({ key }) => localStorage.setItem(key, "second-account"),
    { key: LAST_ACCOUNT_KEY },
  );

  const result = await settled;
  expect(result.newValue).toBe("second-account");
  // This is the trap the guard was rewritten for. By the time the listener runs,
  // reading the key back already returns the NEW value, so a guard that compares
  // the event against localStorage compares a value with itself and never fires.
  expect(result.readBack).toBe("second-account");
  await other.close();
});

test("deleteDatabase makes progress with a live connection open", async ({ page }) => {
  const outcome = await page.evaluate(async () => {
    const name = "unsloth-data-recipes";
    const open = await new Promise<IDBDatabase>((resolve, reject) => {
      const request = indexedDB.open(name, 1);
      request.onupgradeneeded = () => request.result.createObjectStore("rows");
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });

    let blocked = false;
    const deletion = new Promise<string>((resolve) => {
      const request = indexedDB.deleteDatabase(name);
      request.onblocked = () => {
        blocked = true;
        // What the app does next: the reload drops every handle this document holds.
        open.close();
      };
      request.onsuccess = () => resolve("deleted");
      request.onerror = () => resolve("error");
      setTimeout(() => resolve("hung"), 8000);
    });

    const result = await deletion;
    const namesAfter =
      typeof indexedDB.databases === "function"
        ? (await indexedDB.databases()).map((entry) => entry.name)
        : null;
    return { result, blocked, namesAfter };
  });

  expect(
    outcome.result,
    "a delete that never settles would leave one account's recipes for the next",
  ).toBe("deleted");
  if (outcome.namesAfter !== null) {
    expect(outcome.namesAfter).not.toContain("unsloth-data-recipes");
  }
});

test("clearing under an open handle does not throw into the sign-in path", async ({ page }) => {
  const threw = await page.evaluate(async () => {
    await new Promise<void>((resolve) => {
      const request = indexedDB.open("unsloth-data-recipe-executions", 1);
      request.onupgradeneeded = () => request.result.createObjectStore("rows");
      request.onsuccess = () => resolve();
      request.onerror = () => resolve();
    });
    try {
      indexedDB.deleteDatabase("unsloth-data-recipe-executions");
      return false;
    } catch {
      return true;
    }
  });
  expect(threw, "a blocked delete must not refuse the sign-in it runs inside").toBe(false);
});
