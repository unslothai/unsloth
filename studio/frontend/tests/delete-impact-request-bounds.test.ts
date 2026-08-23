// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// An unbounded preview never settles, so the dialog would sit on "Checking whether this model
// can be deleted…" for as long as it stays open: no impact, no "couldn't check" fallback, and
// no re-read of a block that has since lifted, because the poll chain only advances on a
// settled request.

import assert from "node:assert/strict";
import { register } from "node:module";
import { after, test } from "node:test";

register("./helpers/settings-api-resolver.mjs", import.meta.url);

// node:test does not bound a test on its own, and every case here awaits a request that only
// the bound under test can settle, so a regression would hang the suite instead of failing it.
const BOUNDED = { timeout: 5_000 };

const requestedBounds: number[] = [];
let fireTimeout: (() => void) | null = null;
const nativeTimeout = AbortSignal.timeout;
AbortSignal.timeout = ((ms: number) => {
  requestedBounds.push(ms);
  const controller = new AbortController();
  fireTimeout = () =>
    controller.abort(
      new DOMException("The operation timed out.", "TimeoutError"),
    );
  return controller.signal;
}) as typeof AbortSignal.timeout;

let requestSignal: AbortSignal | undefined;
// A request nothing ever answers. `honourSignal` is what fetch itself does; leaving it off
// stands in for the segment fetch cannot cancel — authFetch awaiting a shared session
// refresh on a 401, which carries no signal.
let honourSignal = true;
globalThis.fetch = ((_input: RequestInfo | URL, init?: RequestInit) => {
  requestSignal = init?.signal ?? undefined;
  return new Promise<Response>((_resolve, reject) => {
    if (!honourSignal) {
      return;
    }
    init?.signal?.addEventListener(
      "abort",
      () =>
        reject(new DOMException("The operation was aborted.", "AbortError")),
      { once: true },
    );
  });
}) as typeof fetch;

const { fetchDeleteImpact } = await import(
  "../src/features/hub/inventory/api.ts"
);

after(() => {
  AbortSignal.timeout = nativeTimeout;
});

test(
  "a preview that never answers settles, so the dialog stops saying it is checking",
  BOUNDED,
  async () => {
    honourSignal = true;
    requestedBounds.length = 0;
    const caller = new AbortController();
    const pending = fetchDeleteImpact(
      "org/model",
      "Q4_K_M",
      "/cache/models--org--model/snapshots/rev",
      caller.signal,
    );

    const [bound] = requestedBounds;
    assert.ok(
      bound !== undefined && Number.isFinite(bound) && bound > 0,
      "the preview must carry a timeout, not just the caller's signal",
    );
    assert.notEqual(requestSignal, caller.signal);

    fireTimeout?.();
    // null is what the hook reads as `unavailable`: the dialog warns instead of blocking.
    assert.equal(await pending, null);
    assert.equal(caller.signal.aborted, false);
  },
);

test(
  "a preview sends the cache copy selected by the delete dialog",
  BOUNDED,
  async () => {
    honourSignal = true;
    const original = globalThis.fetch;
    let body: string | undefined;
    globalThis.fetch = ((_input: RequestInfo | URL, init?: RequestInit) => {
      body = String(init?.body);
      return Promise.resolve(
        new Response(
          '{"repo_id":"org/model","reclaimed_bytes":1,"retained_companions":[],"freeable_companions":[],"blocked_by":[]}',
          { status: 200, headers: { "Content-Type": "application/json" } },
        ),
      );
    }) as typeof fetch;
    try {
      await fetchDeleteImpact(
        "org/model",
        null,
        "/cache/models--org--model/snapshots/rev",
      );
      const payload = JSON.parse(body ?? "{}") as Record<string, string>;
      assert.deepEqual(Object.entries(payload).sort(), [
        ["cache_path", "/cache/models--org--model/snapshots/rev"],
        ["repo_id", "org/model"],
      ]);
    } finally {
      globalThis.fetch = original;
    }
  },
);

test(
  "an ambiguous cache response preserves the actionable backend message",
  BOUNDED,
  async () => {
    honourSignal = true;
    const original = globalThis.fetch;
    globalThis.fetch = (() =>
      Promise.resolve(
        new Response(
          JSON.stringify({
            detail: {
              message:
                "Multiple cached copies were found. Choose a cache location to delete.",
              cache_paths: [
                "/cache-a/models--org--model",
                "/cache-b/models--org--model",
              ],
            },
          }),
          { status: 409, headers: { "Content-Type": "application/json" } },
        ),
      )) as typeof fetch;
    try {
      const result = await fetchDeleteImpact("org/model");
      assert.deepEqual(result?.delete_block, {
        status_code: 409,
        detail:
          "Multiple cached copies were found. Choose a cache location to delete.",
        retryable: false,
      });
    } finally {
      globalThis.fetch = original;
    }
  },
);

test(
  "an empty ambiguous cache response uses a clean status fallback",
  BOUNDED,
  async () => {
    honourSignal = true;
    const original = globalThis.fetch;
    globalThis.fetch = (() =>
      Promise.resolve(new Response(null, { status: 409 }))) as typeof fetch;
    try {
      const result = await fetchDeleteImpact("org/model");
      assert.deepEqual(result?.delete_block, {
        status_code: 409,
        detail: "Choose a cache location to delete (409)",
        retryable: false,
      });
    } finally {
      globalThis.fetch = original;
    }
  },
);

test(
  "the bound settles the preview even where the signal cannot reach",
  BOUNDED,
  async () => {
    honourSignal = false;
    const pending = fetchDeleteImpact("org/model", "Q4_K_M");
    fireTimeout?.();
    assert.equal(await pending, null);
  },
);

test(
  "closing the dialog gives up on the preview it started",
  BOUNDED,
  async () => {
    honourSignal = true;
    const caller = new AbortController();
    const pending = fetchDeleteImpact("org/model", null, null, caller.signal);
    caller.abort();
    assert.equal(await pending, null);
  },
);
