// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  type GgufRepoPickHandlers,
  createPickGuard,
  runGgufRepoPick,
} from "../src/lib/diffusion-gguf-pick.ts";

/** A listing whose completion the test controls. */
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}

const flush = () => new Promise((r) => setTimeout(r, 0));

/** Records every side effect a pick is allowed to have, so a stale one can be asserted to have had none. */
function recorder(
  overrides: Partial<Omit<GgufRepoPickHandlers, "load">> & {
    /** What `load` reports back: false is "the load never started". */
    starts?: boolean | (() => boolean);
  } = {},
): { handlers: GgufRepoPickHandlers; log: string[] } {
  const { starts = true, ...rest } = overrides;
  const log: string[] = [];
  const handlers: GgufRepoPickHandlers = {
    resolve: () => Promise.resolve("model-Q4_K_S.gguf"),
    isCurrent: () => true,
    onAmbiguous: () => log.push("ambiguous"),
    onResolved: (filename) => log.push(`resolved:${filename}`),
    onNotStarted: () => log.push("reverted"),
    load: (filename) => {
      log.push(`load:${filename}`);
      return Promise.resolve(typeof starts === "function" ? starts() : starts);
    },
    ...rest,
  };
  return { handlers, log };
}

test("a resolved pick applies its label and loads", async () => {
  const { handlers, log } = recorder();
  assert.equal(await runGgufRepoPick(handlers), true);
  assert.deepEqual(log, [
    "resolved:model-Q4_K_S.gguf",
    "load:model-Q4_K_S.gguf",
  ]);
});

test("an unresolvable pick prompts and loads nothing", async () => {
  const { handlers, log } = recorder({ resolve: () => Promise.resolve(null) });
  assert.equal(await runGgufRepoPick(handlers), false);
  assert.deepEqual(log, ["ambiguous"]);
});

test("a load that never starts takes its own label back", async () => {
  const { handlers, log } = recorder({ starts: false });
  assert.equal(await runGgufRepoPick(handlers), false);
  assert.deepEqual(log, [
    "resolved:model-Q4_K_S.gguf",
    "load:model-Q4_K_S.gguf",
    "reverted",
  ]);
});

test("a superseded pick does nothing at all when its listing lands", async () => {
  // Not even the prompt: it would blame a model the user has already moved on from.
  const listing = deferred<string | null>();
  const { handlers, log } = recorder({
    resolve: () => listing.promise,
    isCurrent: () => false,
  });
  const done = runGgufRepoPick(handlers);
  listing.resolve("model-Q4_K_S.gguf");
  assert.equal(await done, false);
  assert.deepEqual(log, []);
});

test("a superseded pick does not prompt either", async () => {
  const { handlers, log } = recorder({
    resolve: () => Promise.resolve(null),
    isCurrent: () => false,
  });
  assert.equal(await runGgufRepoPick(handlers), false);
  assert.deepEqual(log, []);
});

test("a stale failure does not revert the newer selection's label", async () => {
  // quantRevert is a single slot, so a late rollback would restore this pick's old label over the one that replaced it.
  let current = true;
  const { handlers, log } = recorder({
    isCurrent: () => current,
    // The next pick takes the page while this one's load is in flight, and this load then reports it never started.
    starts: () => {
      current = false;
      return false;
    },
  });
  assert.equal(await runGgufRepoPick(handlers), false);
  assert.deepEqual(log, [
    "resolved:model-Q4_K_S.gguf",
    "load:model-Q4_K_S.gguf",
  ]);
});

test("the newest pick owns the page, whatever order the listings land in", async () => {
  const guard = createPickGuard();
  const listings = [deferred<string | null>(), deferred<string | null>()];
  const log: string[] = [];
  const run = (index: number) => {
    const token = guard.claim();
    return runGgufRepoPick({
      resolve: () => listings[index].promise,
      isCurrent: () => guard.holds(token),
      onAmbiguous: () => log.push(`ambiguous:${index}`),
      onResolved: () => {},
      onNotStarted: () => {},
      load: (filename) => {
        log.push(`load:${index}:${filename}`);
        return Promise.resolve(true);
      },
    });
  };

  const first = run(0);
  await flush();
  const second = run(1);
  await flush();
  // The newer listing lands first, then the stale one.
  listings[1].resolve("b.gguf");
  await flush();
  listings[0].resolve("a.gguf");

  assert.deepEqual(await Promise.all([first, second]), [false, true]);
  assert.deepEqual(log, ["load:1:b.gguf"]);
});

test("releasing the page invalidates the pick holding it", () => {
  // A page switch, an unload or an unmount: nobody owns the page afterwards.
  const guard = createPickGuard();
  const token = guard.claim();
  assert.equal(guard.holds(token), true);
  guard.release();
  assert.equal(guard.holds(token), false);
  // And the token a release lands on is not claimable by an outstanding holder.
  assert.notEqual(guard.claim(), token);
});

test("an eject ends the pick, so a staged download does not come back", () => {
  // Release and cancel differ exactly here: leaving the page defers the staged load, ejecting drops it.
  const guard = createPickGuard();
  const staged = guard.claim();
  guard.cancel();
  assert.equal(guard.holds(staged), false);
  assert.equal(guard.isLatest(staged), false);
  // And the page is claimable again afterwards, with a token of its own.
  const next = guard.claim();
  assert.notEqual(next, staged);
  assert.equal(guard.holds(next), true);
});

test("a release is not a new pick, so a staged download still lands", () => {
  // Leaving the page defers the staged load rather than dropping it; only another pick may take it.
  const guard = createPickGuard();
  const staged = guard.claim();
  guard.release();
  assert.equal(guard.isLatest(staged), true);
  guard.claim();
  assert.equal(guard.isLatest(staged), false);
});
