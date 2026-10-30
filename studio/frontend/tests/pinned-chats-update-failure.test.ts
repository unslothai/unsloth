// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The real pins API and sync runtime under the desktop update screen: a failure the stopped
// backend caused must not toast, however late the queue reports it.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

// See helpers/auth-stub.mjs.
register("./helpers/settings-api-resolver.mjs", import.meta.url);

const { followDesktopUpdateScreen } = await import(
  "../src/lib/desktop-update-activity.ts"
);
const { createChatPinsApi } = await import("../src/features/chat/api/chat-pins-api.ts");
const {
  PinnedChatsBlockedError,
  PinnedChatsValidationError,
  createPinnedChatsSync,
} = await import("../src/features/chat/utils/pinned-chats-sync.ts");
const { isPinFailureUnderDesktopUpdate } = await import(
  "../src/features/chat/utils/pinned-chats-update-failure.ts"
);
const { ChatSettingsRequestError, isRetryablePinError } = await import(
  "../src/features/chat/utils/settings-retry.ts"
);

const noResync = () => Promise.resolve();

// What the real authFetch throws when the backend is unreachable.
const transport = () =>
  Object.assign(new Error("Unsloth isn't running -- please relaunch it."), {
    unslothTransportFailure: true,
  });

function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((r) => (resolve = r));
  return { promise, resolve };
}

function json(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

/** A 200 whose headers arrived but whose body the shutdown cut off. */
function interrupted(): Response {
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(new TextEncoder().encode('{"pinnedIds": ["a"'));
      controller.error(new TypeError("network error"));
    },
  });
  return new Response(body, { status: 200 });
}

type Handler = (url: string, init?: RequestInit) => Promise<Response>;
let handle: Handler = async () => json({});
globalThis.fetch = ((input: RequestInfo | URL, init?: RequestInit) =>
  handle(String(input), init)) as typeof fetch;

/** The pins routes over an in-memory list; `down` makes every request fail like authFetch would. */
function pinsServer() {
  const server = {
    down: false,
    pins: [] as string[],
    // When set, a read while down waits for it before failing.
    holdReads: undefined as Promise<void> | undefined,
    readsWhileDown: 0,
  };
  handle = async (url, init) => {
    if (server.down) {
      if (url.endsWith("/api/chat/pins")) {
        server.readsWhileDown += 1;
        await server.holdReads;
      }
      throw transport();
    }
    if (url.endsWith("/api/chat/pins")) return json({ pinnedIds: server.pins });
    if (url.endsWith("/api/chat/settings/compare-and-set")) {
      const body = JSON.parse(String(init?.body));
      const applied =
        JSON.stringify(body.expected.pinnedChatIds ?? server.pins) ===
        JSON.stringify(server.pins);
      if (applied) server.pins = body.patch.pinnedChatIds;
      return json({ settings: { pinnedChatIds: server.pins }, applied });
    }
    throw new Error(`unexpected ${url}`);
  };
  return server;
}

/** The runtime as sync-pinned-chats wires it, toasting what isPinFailureUnderDesktopUpdate lets through. */
function pinRuntime() {
  const api = createChatPinsApi(new AbortController().signal);
  const raised: unknown[] = [];
  const toasted: unknown[] = [];
  const sync = createPinnedChatsSync({
    load: api.load,
    compareAndSet: api.compareAndSet,
    readLegacy: () => [],
    isRetryable: isRetryablePinError,
    publish: () => {},
    onError: (error) => {
      raised.push(error);
      if (!isPinFailureUnderDesktopUpdate(error)) toasted.push(error);
    },
  });
  return { api, sync, raised, toasted };
}

test("only failures from the stopped backend are pin update failures", () => {
  const failures = [
    transport(),
    // createChatPinsApi aborts a request after 10s.
    new DOMException("The operation was aborted.", "AbortError"),
    new DOMException("The operation timed out.", "TimeoutError"),
    new PinnedChatsBlockedError("An earlier pin change is still waiting to save."),
  ];
  for (const error of failures) assert.equal(isPinFailureUnderDesktopUpdate(error), false);
  const leave = followDesktopUpdateScreen(true, false, noResync);
  try {
    for (const error of failures) assert.equal(isPinFailureUnderDesktopUpdate(error), true);
    // The backend answered: a real rejection still reports.
    assert.equal(
      isPinFailureUnderDesktopUpdate(new ChatSettingsRequestError("Bad", 500, null)),
      false,
    );
    assert.equal(
      isPinFailureUnderDesktopUpdate(new PinnedChatsValidationError("Invalid")),
      false,
    );
  } finally {
    leave();
  }
});

test("a pin read whose body the shutdown cut off is an update failure", async () => {
  const { api } = pinRuntime();
  handle = async () => interrupted();
  const leave = followDesktopUpdateScreen(true, false, noResync);
  let failure: unknown;
  try {
    failure = await api.load().then(() => assert.fail("expected the read to fail"), (e) => e);
  } finally {
    leave();
  }
  assert.equal((failure as { unslothTransportFailure?: boolean }).unslothTransportFailure, true);
  assert.equal(isPinFailureUnderDesktopUpdate(failure), true);
});

test("a pin write whose body the shutdown cut off is an update failure", async () => {
  const { api } = pinRuntime();
  handle = async () => interrupted();
  const leave = followDesktopUpdateScreen(true, false, noResync);
  let failure: unknown;
  try {
    failure = await api
      .compareAndSet([], ["a"])
      .then(() => assert.fail("expected the write to fail"), (e) => e);
  } finally {
    leave();
  }
  assert.equal((failure as { unslothTransportFailure?: boolean }).unslothTransportFailure, true);
  assert.equal(isPinFailureUnderDesktopUpdate(failure), true);
});

test("a malformed pin response still reports under the update screen", async () => {
  const { api } = pinRuntime();
  handle = async () => new Response('{"pinnedIds": [', { status: 200 });
  const leave = followDesktopUpdateScreen(true, false, noResync);
  try {
    const failure = await api.load().then(() => assert.fail("expected the read to fail"), (e) => e);
    assert.ok(failure instanceof SyntaxError);
    assert.equal(isPinFailureUnderDesktopUpdate(failure), false);
  } finally {
    leave();
  }
});

async function queueTwoEditsWhileDown(
  runtime: ReturnType<typeof pinRuntime>,
  server: ReturnType<typeof pinsServer>,
) {
  await runtime.sync.refresh();
  server.down = true;
  return Promise.allSettled([
    runtime.sync.setPinned(["a"], true),
    runtime.sync.setPinned(["b"], true),
  ]);
}

test("pin edits queued behind a failed write stay silent under the update screen", async () => {
  const server = pinsServer();
  const runtime = pinRuntime();
  const leave = followDesktopUpdateScreen(true, false, noResync);
  try {
    await queueTwoEditsWhileDown(runtime, server);
  } finally {
    leave();
  }
  // The second edit waited behind the first, which could not reach the backend.
  assert.ok(runtime.raised.some((error) => error instanceof PinnedChatsBlockedError));
  assert.deepEqual(runtime.toasted, []);
  // Both edits stayed queued and save once the backend is back.
  server.down = false;
  await runtime.sync.refresh();
  assert.deepEqual([...server.pins].sort(), ["a", "b"]);
});

test("a write that fails under the update stays silent when it reports after the screen drops", async () => {
  const server = pinsServer();
  const runtime = pinRuntime();
  const release = deferred();
  server.holdReads = release.promise;
  const leave = followDesktopUpdateScreen(true, false, noResync);
  const settled = queueTwoEditsWhileDown(runtime, server);
  // The failed write's recovery read is in flight when the update screen drops.
  while (server.readsWhileDown === 0) await new Promise((r) => setImmediate(r));
  leave();
  release.resolve();
  await settled;
  assert.equal(runtime.raised.length, 2);
  assert.ok(runtime.raised[1] instanceof PinnedChatsBlockedError);
  assert.deepEqual(runtime.toasted, []);
});

test("the same pin failures still toast outside an update", async () => {
  const server = pinsServer();
  const runtime = pinRuntime();
  await queueTwoEditsWhileDown(runtime, server);
  assert.ok(runtime.raised.length > 0);
  assert.equal(runtime.toasted.length, runtime.raised.length);
});
