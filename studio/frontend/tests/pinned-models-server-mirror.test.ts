// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  registerStoreStubResolver,
} from "./helpers/kit.ts";

registerStoreStubResolver();
const { store } = installLocalStorageFake();

const { hydratePins, pinsMirrorSettledForTests, resetPinsMirrorForTests } =
  await import(
    "../src/features/model-picker/components/model-selector/pins-mirror.ts"
  );
const { usePinnedModelsStore } = await import(
  "../src/features/model-picker/components/model-selector/pinned-models.ts"
);
const { usePinnedConnectedModelsStore } = await import(
  "../src/features/model-picker/components/model-selector/pinned-connected-models.ts"
);
const { setAuthFetchHandler } = await import("./helpers/store-stubs/auth.ts");

const PINNED = "unsloth_pinned_models";
const CONNECTED = "unsloth_pinned_connected_models";

type Server = { pinned: string[] | null; connected: string[] | null };

function serve(server: Server): { puts: Record<string, unknown>[] } {
  const puts: Record<string, unknown>[] = [];
  setAuthFetchHandler((_url, init) => {
    if (init?.method === "PUT") {
      const body = JSON.parse(String(init.body)) as Record<string, unknown>;
      puts.push(body);
      Object.assign(server, body);
    }
    return new Response(JSON.stringify(server), { status: 200 });
  });
  return { puts };
}

function reset(): void {
  store.clear();
  resetPinsMirrorForTests();
  usePinnedModelsStore.setState({ pinned: [] });
  usePinnedConnectedModelsStore.setState({ pinned: [] });
}

// Sends await a dynamic import, so a fixed number of ticks does not drain them on every platform.
const settle = () => pinsMirrorSettledForTests();

test("an account switch's empty browser gets its pins back from the server", async () => {
  reset();
  const { puts } = serve({
    pinned: ["org/a::Q4_K_M", "org/b"],
    connected: ["external::c1::gpt"],
  });
  await hydratePins();
  assert.deepEqual(usePinnedModelsStore.getState().pinned, [
    "org/a::Q4_K_M",
    "org/b",
  ]);
  assert.deepEqual(usePinnedConnectedModelsStore.getState().pinned, [
    "external::c1::gpt",
  ]);
  assert.equal(store.get(PINNED), JSON.stringify(["org/a::Q4_K_M", "org/b"]));
  assert.equal(store.get(CONNECTED), JSON.stringify(["external::c1::gpt"]));
  await settle();
  assert.deepEqual(puts, []);
});

test("an account the server has no pins for is seeded from this browser", async () => {
  reset();
  store.set(PINNED, JSON.stringify(["org/local"]));
  usePinnedModelsStore.setState({ pinned: ["org/local"] });
  const { puts } = serve({ pinned: null, connected: null });
  await hydratePins();
  await settle();
  assert.deepEqual(puts, [{ pinned: ["org/local"] }]);
});

test("a pin made before hydration is held, then wins over the server's list", async () => {
  reset();
  const { puts } = serve({ pinned: ["org/server"], connected: null });
  usePinnedModelsStore.getState().togglePinned("org/new");
  await settle();
  assert.deepEqual(puts, []);
  await hydratePins();
  await settle();
  assert.deepEqual(puts, [{ pinned: ["org/new"] }]);
  assert.deepEqual(usePinnedModelsStore.getState().pinned, ["org/new"]);
});

test("after hydration every edit is mirrored, connected pins included", async () => {
  reset();
  const { puts } = serve({ pinned: [], connected: [] });
  await hydratePins();
  usePinnedModelsStore.getState().togglePinned("org/a", "Q8_0");
  usePinnedConnectedModelsStore
    .getState()
    .togglePinnedConnected("external::c1::gpt");
  await settle();
  assert.deepEqual(puts, [
    { pinned: ["org/a::Q8_0"] },
    { connected: ["external::c1::gpt"] },
  ]);
});

test("a failed read leaves the browser alone and sends nothing", async () => {
  reset();
  store.set(PINNED, JSON.stringify(["org/local"]));
  usePinnedModelsStore.setState({ pinned: ["org/local"] });
  const puts: unknown[] = [];
  setAuthFetchHandler((_url, init) => {
    if (init?.method === "PUT") puts.push(init.body);
    return new Response("{}", { status: 500 });
  });
  await hydratePins();
  usePinnedModelsStore.getState().togglePinned("org/b");
  await settle();
  assert.deepEqual(puts, []);
  assert.equal(store.get(PINNED), JSON.stringify(["org/b", "org/local"]));
  setAuthFetchHandler(null);
});
