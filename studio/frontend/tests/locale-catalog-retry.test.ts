// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Chrome, Edge and Firefox before 155 keep a failed module in the module map
// keyed by URL, so importing the same catalog again resolves to the stored
// failure without a request. Dropping our own in-flight promise only makes the
// store willing to ask again; the ask has to reach the network to be a retry.

import assert from "node:assert/strict";
import test from "node:test";
import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
installLocalStorageFake();

const messagesModule = await import("../src/i18n/messages.ts");

const CHUNK_URL = "https://studio.example/assets/de-a1b2c3.js";

type ImporterCall = { locale: string; retryUrl: string | null };

/** Runs a load to completion, reporting whether it failed. */
async function attempt(load: Promise<void> | undefined): Promise<string> {
  const [outcome] = await Promise.allSettled([load]);
  return outcome.status;
}

function fetchFailure(url: string): TypeError {
  return new TypeError(`Failed to fetch dynamically imported module: ${url}`);
}

test("a failed catalog is retried from a different URL", async () => {
  const calls: ImporterCall[] = [];
  const importer = (locale: string, retryUrl: string | null) => {
    calls.push({ locale, retryUrl });
    if (calls.length === 1) return Promise.reject(fetchFailure(CHUNK_URL));
    return Promise.resolve({ de: { common: { cancel: "Abbrechen" } } });
  };

  const first = await attempt(
    messagesModule.loadLocaleMessages("de", importer),
  );
  const second = await attempt(
    messagesModule.loadLocaleMessages("de", importer),
  );

  assert.equal(calls.length, 2);
  assert.equal(first, "rejected");
  assert.equal(second, "fulfilled");
  // The first load is a plain import, so the happy path keeps the normal
  // caching of its hashed file, and only the retry carries a one-off query.
  assert.equal(calls[0]?.retryUrl, null);
  assert.notEqual(calls[1]?.retryUrl, null);
  assert.notEqual(calls[1]?.retryUrl, CHUNK_URL);
  const retried = new URL(calls[1]?.retryUrl ?? "");
  assert.equal(retried.origin + retried.pathname, CHUNK_URL);
  assert.match(
    retried.searchParams.get(messagesModule.CATALOG_RETRY_PARAM) ?? "",
    /^\d+$/,
  );
  assert.equal(
    messagesModule.translate("common.cancel", undefined, "de"),
    "Abbrechen",
  );
});

test("a catalog that loaded is not asked for again", () => {
  assert.equal(
    messagesModule.loadLocaleMessages("de", () => {
      throw new Error("should not import");
    }),
    undefined,
  );
});

test("a failure with no URL in it leaves nothing to cache-bust", async () => {
  const calls: ImporterCall[] = [];
  const importer = (locale: string, retryUrl: string | null) => {
    calls.push({ locale, retryUrl });
    if (calls.length === 1) return Promise.reject(new Error("boom"));
    return Promise.resolve({ ru: { common: { cancel: "Отмена" } } });
  };

  await attempt(messagesModule.loadLocaleMessages("ru", importer));
  await attempt(messagesModule.loadLocaleMessages("ru", importer));

  assert.deepEqual(
    calls.map((call) => call.retryUrl),
    [null, null],
  );
});

test("a retry that fails again gets a fresh URL rather than the failed one", async () => {
  const calls: ImporterCall[] = [];
  const importer = (locale: string, retryUrl: string | null) => {
    calls.push({ locale, retryUrl });
    if (calls.length < 3) return Promise.reject(fetchFailure(CHUNK_URL));
    return Promise.resolve({ it: { common: { cancel: "Annulla" } } });
  };

  await attempt(messagesModule.loadLocaleMessages("it", importer));
  await attempt(messagesModule.loadLocaleMessages("it", importer));
  await attempt(messagesModule.loadLocaleMessages("it", importer));

  assert.equal(calls.length, 3);
  assert.equal(calls[0]?.retryUrl, null);
  assert.notEqual(calls[1]?.retryUrl, calls[2]?.retryUrl);
  // One query, so a repeated failure cannot grow the URL.
  assert.equal(
    [...new URL(calls[2]?.retryUrl ?? "").searchParams.keys()].length,
    1,
  );
});
