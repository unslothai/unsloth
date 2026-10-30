// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  followDesktopUpdateScreen,
  isBackendDownForDesktopUpdate,
  isSilencedDesktopUpdateFailure,
} from "../src/lib/desktop-update-activity.ts";
import {
  PinnedChatsBlockedError,
  PinnedChatsValidationError,
  createPinnedChatsSync,
} from "../src/features/chat/utils/pinned-chats-sync.ts";
import { isPinFailureUnderDesktopUpdate } from "../src/features/chat/utils/pinned-chats-update-failure.ts";
import {
  ChatSettingsRequestError,
  isRetryablePinError,
} from "../src/features/chat/utils/settings-retry.ts";
import { readSrc } from "./helpers/kit.ts";

const RUNTIME = readSrc("features/chat/hooks/use-chat-model-runtime.ts");
const RAG = readSrc("features/rag/components/use-rag-documents.ts");
const PIN_HOOK = readSrc("features/chat/hooks/use-pinned-chats-sync.ts");
const PIN_SYNC = readSrc("features/chat/sync-pinned-chats.ts");

function region(source: string, from: string, to: string): string {
  const start = source.indexOf(from);
  const end = source.indexOf(to, start);
  assert.ok(start !== -1, `region start not found: ${from}`);
  assert.ok(end > start, `region end not found after start: ${to}`);
  return source.slice(start, end);
}

const transport = () =>
  Object.assign(new Error("Unsloth isn't running -- please relaunch it."), {
    unslothTransportFailure: true,
  });

function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((r) => (resolve = r));
  return { promise, resolve };
}

const noResync = () => Promise.resolve();
const settle = () => new Promise((r) => setImmediate(r));

test("only a transport failure under the update screen is silenced", () => {
  assert.equal(isSilencedDesktopUpdateFailure(transport(), false), false);
  const cleanup = followDesktopUpdateScreen(true, false, noResync);
  assert.equal(isSilencedDesktopUpdateFailure(transport(), false), true);
  assert.equal(
    isSilencedDesktopUpdateFailure(new Error("Internal Server Error"), true),
    false,
  );
  cleanup();
  assert.equal(isBackendDownForDesktopUpdate(), false);
});

test("a read issued under the update screen stays silent after the flag drops", () => {
  const cleanup = followDesktopUpdateScreen(true, false, noResync);
  const downWhenIssued = isBackendDownForDesktopUpdate();
  cleanup();
  assert.equal(isSilencedDesktopUpdateFailure(transport(), downWhenIssued), true);
});

test("leaving the update screen holds the flag until the resync settles", async () => {
  followDesktopUpdateScreen(true, false, noResync)();
  const resync = deferred();
  let resyncs = 0;
  followDesktopUpdateScreen(false, true, () => {
    resyncs += 1;
    return resync.promise;
  });
  assert.equal(resyncs, 1);
  assert.equal(isBackendDownForDesktopUpdate(), true);
  resync.resolve();
  await settle();
  assert.equal(isBackendDownForDesktopUpdate(), false);
});

test("a retry during the resync keeps the flag when the resync settles", async () => {
  const resync = deferred();
  const leave = followDesktopUpdateScreen(false, true, () => resync.promise);
  leave();
  const retry = followDesktopUpdateScreen(true, false, noResync);
  resync.resolve();
  await settle();
  assert.equal(isBackendDownForDesktopUpdate(), true);
  retry();
});

test("no resync without a transition off the update screen", () => {
  let resyncs = 0;
  const count = () => {
    resyncs += 1;
    return Promise.resolve();
  };
  followDesktopUpdateScreen(false, false, count)();
  followDesktopUpdateScreen(true, false, count)();
  followDesktopUpdateScreen(true, true, count)();
  assert.equal(resyncs, 0);
});

test("the chat status refresh latches the flag at issue and checks it before reporting", () => {
  const sync = region(
    RUNTIME,
    "async function syncInferenceStatusToStore(",
    "async function refreshAndWaitForServerModel(",
  );
  const catchAt = sync.indexOf("} catch (error) {");
  const latch = sync.indexOf("const downWhenIssued = isBackendDownForDesktopUpdate();");
  assert.ok(latch !== -1 && latch < catchAt);
  const guard = sync.indexOf(
    "if (isSilencedDesktopUpdateFailure(error, downWhenIssued)) return;",
  );
  assert.ok(guard > catchAt);
  assert.ok(guard < sync.indexOf("setModelsError(message)"));
  assert.ok(guard < sync.indexOf("toast.error"));
});

test("the documents refresh latches the flag at issue and checks it before reporting", () => {
  const refresh = region(RAG, "const refresh = useCallback(", "const loadProjectSources");
  const latch = refresh.indexOf("const downWhenIssued = isBackendDownForDesktopUpdate();");
  const guard = refresh.indexOf(
    "if (isSilencedDesktopUpdateFailure(err, downWhenIssued)) return false;",
  );
  assert.ok(latch !== -1 && latch < refresh.indexOf("} catch (err) {"));
  assert.ok(guard !== -1 && guard < refresh.indexOf("toast.error"));
});

test("the pin sync poll stands down under the update screen", () => {
  const refresh = region(PIN_HOOK, "const refresh = () => {", "const reconcileSession");
  const gate = refresh.indexOf("if (isBackendDownForDesktopUpdate()) return;");
  assert.ok(gate !== -1 && gate < refresh.indexOf("loadRuntime()"));
});

test("the pin sync error toast checks for an update failure first", () => {
  const onError = region(PIN_SYNC, "onError: (error) => {", "// Every delete path");
  const guard = onError.indexOf("if (isPinFailureUnderDesktopUpdate(error)) return;");
  assert.ok(guard !== -1 && guard < onError.indexOf("toast.error"));
});

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

/** The real sync runtime over a backend that `up` takes down, toasting as sync-pinned-chats does. */
function pinRuntime() {
  const backend = { up: true, value: [] as string[] };
  const raised: unknown[] = [];
  const toasted: unknown[] = [];
  const sync = createPinnedChatsSync({
    load: async () => {
      if (!backend.up) throw transport();
      return backend.value.slice();
    },
    compareAndSet: async (expected, next) => {
      if (!backend.up) throw transport();
      const applied = JSON.stringify(backend.value) === JSON.stringify(expected);
      if (applied) backend.value = next.slice();
      return { applied, pinnedIds: backend.value.slice() };
    },
    readLegacy: () => [],
    isRetryable: isRetryablePinError,
    publish: () => {},
    onError: (error) => {
      raised.push(error);
      if (!isPinFailureUnderDesktopUpdate(error)) toasted.push(error);
    },
  });
  return { backend, sync, raised, toasted };
}

async function queueTwoEditsWhileDown(runtime: ReturnType<typeof pinRuntime>) {
  await runtime.sync.refresh();
  runtime.backend.up = false;
  await Promise.allSettled([
    runtime.sync.setPinned(["a"], true),
    runtime.sync.setPinned(["b"], true),
  ]);
  // The second edit waits behind the first, which could not reach the backend.
  assert.ok(runtime.raised.some((error) => error instanceof PinnedChatsBlockedError));
}

test("pin edits queued behind a failed write stay silent under the update screen", async () => {
  const runtime = pinRuntime();
  const leave = followDesktopUpdateScreen(true, false, noResync);
  try {
    await queueTwoEditsWhileDown(runtime);
    assert.deepEqual(runtime.toasted, []);
  } finally {
    leave();
  }
  // Both edits stayed queued and save once the backend is back.
  runtime.backend.up = true;
  await runtime.sync.refresh();
  assert.deepEqual([...runtime.backend.value].sort(), ["a", "b"]);
});

test("the same pin failures still toast outside an update", async () => {
  const runtime = pinRuntime();
  await queueTwoEditsWhileDown(runtime);
  assert.equal(runtime.toasted.length, runtime.raised.length);
});

test("the update layer drives the flag and resyncs chat when the screen drops", () => {
  const layer = region(
    readSrc("app/provider.tsx"),
    "function TauriUpdateLayer(",
    "const HIDDEN_TITLEBAR_SIDEBAR_ROUTES",
  );
  assert.match(
    layer,
    /const wasUpdating = wasUpdatingRef\.current;\s*wasUpdatingRef\.current = isUpdating;[\s\S]*?return followDesktopUpdateScreen\(\s*isUpdating,\s*wasUpdating,\s*resyncInferenceStatusAfterServerModelChange,\s*\);\s*\}, \[isUpdating\]\);/,
  );
});
