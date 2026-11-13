// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { register } from "node:module";

register("./helpers/toast-resolver.mjs", import.meta.url);

const { calls } = await import("./helpers/toast-stub.mjs");
const { ejectWithToast } = await import("../src/lib/media-eject.ts");
const { readSrc } = await import("./helpers/kit.ts");

type Call = { kind: string; title?: string; id?: string; options?: { id?: string } };

function deferred() {
  let resolve!: (v: boolean) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<boolean>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function shown(): [string, string | undefined][] {
  return (calls as Call[]).map((c) => [c.kind, c.title ?? c.id]);
}

test("eject shows Unloading model, then Model unloaded on the same toast", async () => {
  calls.length = 0;
  await ejectWithToast({ current: false }, async () => true);
  const success = (calls as Call[])[1];
  assert.deepEqual(shown(), [
    ["loading", "Unloading model"],
    ["success", "Model unloaded"],
  ]);
  assert.equal(success.options?.id, "loading-1");
});

test("a second eject while unloading sends no unload and says to wait", async () => {
  calls.length = 0;
  const inFlight = { current: false };
  const gate = deferred();
  let unloads = 0;
  const unload = () => {
    unloads++;
    return gate.promise;
  };
  const first = ejectWithToast(inFlight, unload);
  const second = ejectWithToast(inFlight, unload);
  const unloadsDuring = unloads;
  const last = shown().at(-1);
  gate.resolve(true);
  await Promise.all([first, second]);
  assert.equal(unloadsDuring, 1);
  assert.deepEqual(last, ["info", "Wait for the model to finish unloading."]);
  calls.length = 0;
  await ejectWithToast(inFlight, unload);
  assert.equal(unloads, 2, "the guard clears once the unload ends");
});

test("a failed or still-loading unload drops the loading toast", async () => {
  calls.length = 0;
  const inFlight = { current: false };
  await ejectWithToast(inFlight, async () => false);
  assert.deepEqual(shown(), [
    ["loading", "Unloading model"],
    ["dismiss", "loading-1"],
  ]);
  calls.length = 0;
  await assert.rejects(
    ejectWithToast(inFlight, async () => {
      throw new Error("boom");
    }),
  );
  assert.equal(shown().at(-1)?.[0], "dismiss");
  assert.equal(inFlight.current, false);
});

for (const path of ["features/images/images-page.tsx", "features/video/video-page.tsx"]) {
  test(`${path} routes the pill eject through ejectWithToast`, () => {
    const page = readSrc(path);
    assert.match(page, /onEject=\{status\?\.loaded && busy !== "unloading" \? handleEject : undefined\}/);
    assert.match(page, /const handleEject = useCallback\(\(\) => ejectWithToast\(ejecting, handleUnload\), \[handleUnload\]\);/);
  });
}
