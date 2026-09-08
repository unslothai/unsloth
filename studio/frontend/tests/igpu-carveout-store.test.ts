// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Every model load calls showCarveoutAdvice, including the loads that carry no
// advice at all. The store is therefore where one model's numbers could outlive
// the model they describe, and where a dismissal could report an allocation the
// user is no longer running.

import assert from "node:assert/strict";
import test from "node:test";

import { registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();

const { setAuthFetchHandler } = await import("./helpers/store-stubs/auth.ts");
const { useIgpuCarveoutDialogStore } = await import(
  "../src/features/igpu-carveout/stores/igpu-carveout-dialog-store.ts"
);

const ADVICE = {
  current_gb: 32,
  needed_gb: 42.9,
  suggested_gb: 48,
  machine_gb: 127.8,
  host_left_gb: 79.8,
  message: "This model's weights are about 43 GB...",
};

function reset() {
  useIgpuCarveoutDialogStore.setState({ open: false, advice: null });
  setAuthFetchHandler(null);
}

const store = () => useIgpuCarveoutDialogStore.getState();

test("advice opens the dialog", () => {
  reset();
  store().show(ADVICE);
  assert.equal(store().open, true);
  assert.equal(store().advice?.suggested_gb, 48);
});

test("a malformed payload opens nothing", () => {
  reset();
  for (const bad of [null, undefined, {}, "advice", { ...ADVICE, message: "" }]) {
    store().show(bad);
    assert.equal(store().open, false);
  }
});

test("close keeps the advice long enough to animate out", () => {
  // The dialog renders null without advice, and the shared AlertDialog carries a
  // 100ms data-closed:animate-out. Clearing on close would blank the text
  // mid-animation instead of fading it.
  reset();
  store().show(ADVICE);
  store().close();
  assert.equal(store().open, false);
  assert.ok(store().advice, "advice must survive the exit animation");
});

test("a later load carrying no advice clears the previous load's numbers", () => {
  // This is the moment the old figures stop being true. Without it the store
  // holds model A's numbers after model B has loaded, and anything added later
  // that reads them outside the `open` guard reads the wrong machine.
  reset();
  store().show(ADVICE);
  store().close();
  store().show(undefined);
  assert.equal(store().advice, null);
});

test("a second advice does not replace the text under the user", () => {
  reset();
  store().show(ADVICE);
  store().show({ ...ADVICE, suggested_gb: 96 });
  assert.equal(store().advice?.suggested_gb, 48);
});

test("remind me later sends nothing", async () => {
  reset();
  let called = false;
  setAuthFetchHandler(() => {
    called = true;
    return new Response("{}", { status: 200 });
  });
  store().show(ADVICE);
  store().close();
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(called, false);
});

test("dismissing posts the allocation it was dismissed at", async () => {
  reset();
  const seen: Array<{ url: string; body: unknown; method?: string }> = [];
  setAuthFetchHandler((url, init) => {
    seen.push({
      url,
      method: init?.method,
      body: init?.body ? JSON.parse(String(init.body)) : null,
    });
    return new Response(JSON.stringify({ dismissed_at_gb: 32 }), { status: 200 });
  });
  store().show(ADVICE);
  store().dismissForever();
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(seen.length, 1);
  assert.equal(seen[0].method, "POST");
  assert.match(seen[0].url, /igpu-carveout-notice\/dismiss/);
  assert.deepEqual(seen[0].body, { current_gb: 32 });
});

test("a failing dismissal is swallowed rather than rejected", async () => {
  // The dialog is already gone by then; the worst case is the notice returning
  // on a later load, which is much better than an unhandled rejection.
  reset();
  setAuthFetchHandler(() => {
    throw new TypeError("network down");
  });
  store().show(ADVICE);
  store().dismissForever();
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(store().open, false);
});
