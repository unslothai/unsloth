// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The response carries runtime state that goes stale as soon as a model is
// loaded, so the module coalesces concurrent reads but never caches. Saving it
// must also drop the auto-switch cache, whose idle TTL residency vetoes.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

// The settings API modules reach authFetch through the auth barrel, which
// re-exports login-page.tsx. See helpers/auth-stub.mjs.
register("./helpers/settings-api-resolver.mjs", import.meta.url);
installLocalStorageFake();

type Listener = (event: Event) => void;
const listeners = new Map<string, Set<Listener>>();
Object.assign(globalThis.window as object, {
  addEventListener: (type: string, fn: Listener) => {
    if (!listeners.has(type)) listeners.set(type, new Set());
    listeners.get(type)?.add(fn);
  },
  removeEventListener: (type: string, fn: Listener) => {
    listeners.get(type)?.delete(fn);
  },
  dispatchEvent: (event: Event) => {
    for (const fn of listeners.get(event.type) ?? []) fn(event);
    return true;
  },
});

const API = {
  keep_resident: false,
  no_ram_reserve: false,
  default_keep_resident: false,
  default_no_ram_reserve: false,
  mlock_active: false,
  reload_required: false,
  memlock_limit_bytes: null as number | null,
};

let calls: string[] = [];
let nextBody: Record<string, unknown> = { ...API };
let release: (() => void) | null = null;

globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
  const url = String(
    typeof input === "string" ? input : (input as Request).url,
  );
  calls.push(`${init?.method ?? "GET"} ${url}`);
  const body = { ...nextBody };
  if (release) {
    await new Promise<void>((resolve) => {
      release = resolve;
    });
  }
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}) as typeof fetch;

const {
  loadModelMemorySettings,
  subscribeModelMemorySettings,
  updateModelMemorySettings,
} = await import("../src/features/settings/api/model-memory.ts");
const autoSwitch = await import(
  "../src/features/settings/api/openai-auto-switch.ts"
);

test("concurrent reads share one request", async () => {
  calls = [];
  const [a, b, c] = await Promise.all([
    loadModelMemorySettings(),
    loadModelMemorySettings(),
    loadModelMemorySettings(),
  ]);
  assert.equal(calls.length, 1, "three callers, one GET");
  assert.deepEqual(a, b);
  assert.deepEqual(b, c);
});

test("a 404 is an absent route, not a failed read", async () => {
  // The resident-model shortcut treats the two oppositely: an older backend has no such
  // setting to disagree about, while a read that could not be made says nothing, and
  // assuming it said no is how a saved policy goes missing.
  calls = [];
  const original = globalThis.fetch;
  globalThis.fetch = (async () =>
    new Response("{}", { status: 404 })) as typeof fetch;
  await assert.rejects(
    loadModelMemorySettings({ force: true }),
    (error: Error) => {
      assert.equal(error.name, "SettingsRouteAbsentError");
      return true;
    },
  );
  globalThis.fetch = (async () =>
    new Response("boom", { status: 503 })) as typeof fetch;
  await assert.rejects(
    loadModelMemorySettings({ force: true }),
    (error: Error) => {
      assert.notEqual(error.name, "SettingsRouteAbsentError");
      return true;
    },
  );
  globalThis.fetch = original;
});

test("a forced read does not join one already in flight", async () => {
  // Sharing is right for two panels painting the same answer and wrong for the
  // resident-model shortcut: a read that started before a policy save describes the
  // policy it replaced, and a reloadRequired false from that would suppress the very
  // load the save was made for.
  calls = [];
  const joined = loadModelMemorySettings();
  // The fake snapshots the body when the request goes out, so the second GET carries the
  // saved policy and the first still carries the one it replaced.
  nextBody = { ...API, reload_required: true };
  const forced = loadModelMemorySettings({ force: true });
  assert.equal(calls.length, 2, "the forced read must issue its own GET");
  const [stale, fresh] = await Promise.all([joined, forced]);
  assert.equal(
    stale.reloadRequired,
    false,
    "the shared read keeps its own answer",
  );
  assert.equal(
    fresh.reloadRequired,
    true,
    "the forced read sees the saved policy",
  );
  nextBody = { ...API };
});

test("a displaced read neither publishes nor frees the slot", async () => {
  // Forcing replaces an in-flight read, and that older request is still running. It
  // describes the state its replacement was issued because of, so it must not repaint
  // subscribers, and it must not clear a sharing handle it no longer owns.
  const original = globalThis.fetch;
  const pending: ((body: Record<string, unknown>) => void)[] = [];
  let issued = 0;
  globalThis.fetch = (async () => {
    issued += 1;
    return new Promise<Response>((resolve) => {
      pending.push((body) =>
        resolve(
          new Response(JSON.stringify(body), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          }),
        ),
      );
    });
  }) as typeof fetch;

  const published: boolean[] = [];
  const stop = subscribeModelMemorySettings((settings) => {
    published.push(settings.reloadRequired);
  });
  try {
    const displaced = loadModelMemorySettings();
    const forced = loadModelMemorySettings({ force: true });
    assert.equal(issued, 2);

    // The displaced request lands first, while its replacement is still in flight.
    pending[0]({ ...API, reload_required: false });
    assert.equal(
      await displaced,
      await displaced,
      "its own caller still gets an answer",
    );
    await Promise.resolve();
    assert.deepEqual(
      published,
      [],
      "a superseded read must not repaint subscribers",
    );

    // The slot still belongs to the forced read, so a new caller joins it.
    const joiner = loadModelMemorySettings();
    assert.equal(issued, 2, "the displaced read freed a slot it did not own");

    pending[1]({ ...API, reload_required: true });
    assert.equal((await forced).reloadRequired, true);
    assert.equal((await joiner).reloadRequired, true);
    assert.deepEqual(
      published,
      [true],
      "only the current read speaks for everyone",
    );
  } finally {
    stop();
    globalThis.fetch = original;
  }
});

test("a later read is NOT served from a cache", async () => {
  calls = [];
  await loadModelMemorySettings();
  nextBody = { ...API, reload_required: true };
  const second = await loadModelMemorySettings();
  assert.equal(
    calls.length,
    2,
    "runtime state must be refetched, never cached",
  );
  assert.equal(second.reloadRequired, true);
  nextBody = { ...API };
});

test("a failed read does not wedge the in-flight slot", async () => {
  calls = [];
  const original = globalThis.fetch;
  globalThis.fetch = (async () => {
    throw new Error("offline");
  }) as typeof fetch;
  await assert.rejects(loadModelMemorySettings());
  globalThis.fetch = original;
  // The next caller must get a fresh request rather than the rejected promise.
  const after = await loadModelMemorySettings();
  assert.equal(after.keepResident, false);
});

test("only the fields actually set are sent, so the switches save independently", async () => {
  calls = [];
  const bodies: string[] = [];
  const original = globalThis.fetch;
  globalThis.fetch = (async (_input: RequestInfo | URL, init?: RequestInit) => {
    bodies.push(String(init?.body ?? ""));
    return new Response(JSON.stringify(API), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }) as typeof fetch;
  await updateModelMemorySettings({ keepResident: true });
  globalThis.fetch = original;
  assert.deepEqual(JSON.parse(bodies[0] ?? "{}"), { keep_resident: true });
});

test("subscribers receive every published value", async () => {
  const seen: boolean[] = [];
  const stop = subscribeModelMemorySettings((s) => seen.push(s.keepResident));
  nextBody = { ...API, keep_resident: true };
  await loadModelMemorySettings();
  stop();
  nextBody = { ...API };
  await loadModelMemorySettings();
  assert.deepEqual(
    seen,
    [true],
    "one while subscribed, none after unsubscribe",
  );
});

test("saving model memory drops the auto-switch cache", async () => {
  // idle_unload_active is vetoed by residency, so the other endpoint's cached
  // copy is wrong the moment this one is written. hub-page reads it on every
  // status poll, so a stale value survives for the life of the page.
  const first = await autoSwitch.loadOpenAIAutoSwitchSettings();
  assert.ok(first);
  calls = [];
  await autoSwitch.loadOpenAIAutoSwitchSettings();
  assert.equal(calls.length, 0, "auto-switch does cache");

  await updateModelMemorySettings({ keepResident: true });
  calls = [];
  await autoSwitch.loadOpenAIAutoSwitchSettings();
  assert.equal(calls.length, 1, "the write must have invalidated it");
});

test("a read invalidated in flight returns the post-write value, not the stale one", async () => {
  // hub-page puts this straight into idleUnloadArmed, so handing back a
  // response that predates the write is as bad as caching it.
  autoSwitch.invalidateOpenAIAutoSwitchSettings();
  nextBody = { ...API, idle_unload_active: true };
  release = () => {};
  const pending = autoSwitch.loadOpenAIAutoSwitchSettings();

  // Land the invalidation, and the new value, while that response is in flight.
  autoSwitch.invalidateOpenAIAutoSwitchSettings();
  nextBody = { ...API, idle_unload_active: false };
  const resume = release;
  release = null;
  resume?.();

  const settings = await pending;
  assert.equal(
    settings.idleUnloadActive,
    false,
    "the read must have been retried against the new generation",
  );

  // And the retry's value is the one that got cached.
  calls = [];
  const again = await autoSwitch.loadOpenAIAutoSwitchSettings();
  assert.equal(calls.length, 0);
  assert.equal(again.idleUnloadActive, false);
});

test("a caller arriving after the invalidation does not adopt the pre-write read", async () => {
  // The retry above covers the caller that started before the write. One that
  // arrives after it must not share that same GET either: the reply predates the
  // write, and the hub poll puts it straight into idleUnloadArmed, where
  // "disarmed" clears the user's selected checkpoint.
  autoSwitch.invalidateOpenAIAutoSwitchSettings();
  nextBody = { ...API, idle_unload_active: true };
  release = () => {};
  const before = autoSwitch.loadOpenAIAutoSwitchSettings();
  const resume = release;
  release = null;

  autoSwitch.invalidateOpenAIAutoSwitchSettings();
  nextBody = { ...API, idle_unload_active: false };
  const after = autoSwitch.loadOpenAIAutoSwitchSettings();
  resume?.();

  assert.equal(
    (await after).idleUnloadActive,
    false,
    "the post-write caller must read the post-write value",
  );
  assert.equal((await before).idleUnloadActive, false);
  nextBody = { ...API };
});
