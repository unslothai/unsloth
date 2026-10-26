// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Voice Settings "Load into" against the server copy that device-less STT API loads read (#15280).

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

register("./helpers/settings-api-resolver.mjs", import.meta.url);
// stt-device.ts takes authFetch from the module, not the barrel the shared resolver stubs.
const STUB = new URL("./helpers/auth-stub.mjs", import.meta.url).href;
register(
  `data:text/javascript,export function resolve(s, c, n) { return n(s === "@/features/auth/api" ? ${JSON.stringify(STUB)} : s, c); }`,
);

type Call = { method: string; device?: string };

function fakeServer(state: { device: string; saved: boolean }) {
  const calls: Call[] = [];
  globalThis.fetch = (async (_url: string, init?: RequestInit) => {
    const method = init?.method ?? "GET";
    if (method === "PUT") {
      const { device } = JSON.parse(String(init?.body));
      calls.push({ method, device });
      Object.assign(state, { device, saved: true });
    } else {
      calls.push({ method });
    }
    return new Response(JSON.stringify(state), { status: 200 });
  }) as typeof fetch;
  return calls;
}

const { reconcileSttDevicePreference, syncSttDevicePreference } = await import(
  "../src/features/settings/api/stt-device.ts"
);

async function settle() {
  for (let i = 0; i < 10; i++) await new Promise((r) => setTimeout(r, 0));
}

test("a saved server choice wins over this browser's and is not sent back", async () => {
  for (const [server, local] of [
    ["cpu", "auto"], // another browser chose CPU
    ["auto", "cpu"], // a stale tab still holds CPU
  ] as const) {
    const calls = fakeServer({ device: server, saved: true });
    let adopted: string | null = null;
    await reconcileSttDevicePreference(local, (d) => {
      adopted = d;
    });
    await settle();
    assert.equal(adopted, server);
    assert.deepEqual(calls, [{ method: "GET" }]);
  }
});

test("with nothing saved, a non-default local choice migrates and Auto sends nothing", async () => {
  let calls = fakeServer({ device: "auto", saved: false });
  await reconcileSttDevicePreference("cpu", () => assert.fail("nothing to adopt"));
  await settle();
  assert.deepEqual(calls, [{ method: "GET" }, { method: "PUT", device: "cpu" }]);

  calls = fakeServer({ device: "auto", saved: false });
  await reconcileSttDevicePreference("auto", () => assert.fail("nothing to adopt"));
  await settle();
  assert.deepEqual(calls, [{ method: "GET" }]);
});

test("quick changes reach the server in order", async () => {
  const state = { device: "cpu", saved: true };
  const calls: Call[] = [];
  const resolvers: (() => void)[] = [];
  globalThis.fetch = (async (_url: string, init?: RequestInit) => {
    const { device } = JSON.parse(String(init?.body));
    calls.push({ method: "PUT", device });
    // The first save is slow: without ordering the second would land first.
    if (calls.length === 1) await new Promise<void>((r) => resolvers.push(r));
    state.device = device;
    return new Response(JSON.stringify(state), { status: 200 });
  }) as typeof fetch;
  syncSttDevicePreference("auto");
  syncSttDevicePreference("cpu");
  await settle();
  assert.equal(calls.length, 1);
  resolvers[0]();
  await settle();
  assert.deepEqual(calls.map((c) => c.device), ["auto", "cpu"]);
  assert.equal(state.device, "cpu");
});

test("a choice made while the sign-in read is in flight is not undone", async () => {
  for (const saved of [true, false]) {
    const state = { device: "auto", saved };
    const calls: Call[] = [];
    let release: () => void = () => {};
    globalThis.fetch = (async (_url: string, init?: RequestInit) => {
      const method = init?.method ?? "GET";
      if (method === "GET") {
        const snapshot = JSON.stringify(state);
        calls.push({ method });
        await new Promise<void>((r) => {
          release = r;
        });
        return new Response(snapshot, { status: 200 });
      }
      const { device } = JSON.parse(String(init?.body));
      calls.push({ method, device });
      Object.assign(state, { device, saved: true });
      return new Response(JSON.stringify(state), { status: 200 });
    }) as typeof fetch;
    let adopted: string | null = null;
    const done = reconcileSttDevicePreference("cpu", (d) => {
      adopted = d;
    });
    await settle();
    syncSttDevicePreference(saved ? "cpu" : "auto"); // the user picks while the read is out
    await settle();
    release();
    await done;
    await settle();
    assert.equal(adopted, null);
    assert.equal(state.device, saved ? "cpu" : "auto");
  }
});
