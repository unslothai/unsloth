// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The one toast a download start raises, and the three ways it goes away: the
// transfer ends, the user leaves the surface, or the caller never wanted it alone.

import assert from "node:assert/strict";
import test from "node:test";
import { register } from "node:module";

register("./helpers/toast-resolver.mjs", import.meta.url);

let route = "/chat";
Object.defineProperty(globalThis, "window", {
  configurable: true,
  value: {
    location: {
      get pathname() {
        return route;
      },
    },
  },
});

const { calls } = await import("./helpers/toast-stub.mjs");
const {
  dismissStartToast,
  dismissStartToasts,
  liveCallerToast,
  showCallerToast,
  showStartToast,
  startToastId,
} = await import("../src/features/hub/download-manager/start-toast.ts");

function reset(at = "/chat") {
  calls.length = 0;
  route = at;
  dismissStartToasts();
  calls.length = 0;
}

// Compared through a copy, never `calls` itself: node's assert types are `asserts
// actual is T`, so `deepEqual(calls, [])` narrows the shared array to never[] and every
// later read of it fails to typecheck.
function raised() {
  return calls.slice();
}

const MESSAGE = {
  title: "Download is running",
  description: "Nothing is stuck.",
};

test("a start toast is keyed on its job so finalize can drop it", () => {
  reset();
  showStartToast("model:repo:q4", MESSAGE);
  assert.equal(calls[0].kind, "info");
  assert.equal(calls[0].options?.id, startToastId("model:repo:q4"));

  dismissStartToast("model:repo:q4");
  assert.deepEqual(raised()[1], {
    kind: "dismiss",
    id: startToastId("model:repo:q4"),
  });
});

test("leaving the surface that raised it drops it", () => {
  // It lasts 8s from a root-level Toaster, so a start in chat followed by a click
  // on Models parks the composed form over the hub toolbar.
  reset("/chat");
  showStartToast("model:repo:q4", MESSAGE);
  calls.length = 0;

  route = "/hub";
  dismissStartToasts();
  assert.deepEqual(raised(), [
    { kind: "dismiss", id: startToastId("model:repo:q4") },
  ]);

  // Gone for good: a second navigation has nothing left to dismiss.
  calls.length = 0;
  route = "/settings";
  dismissStartToasts();
  assert.deepEqual(raised(), []);
});

test("a raise that lands after the user left is dropped", () => {
  // The Xet reservation is a round trip, so the raise can happen after navigation.
  // The route-change sweep has already run by then, and recording the destination
  // as the origin would leave the composed form on the hub toolbar for its full 8s.
  reset("/chat");
  const startedOn = "/chat";
  route = "/hub";
  showStartToast("model:repo:q4", MESSAGE, startedOn);
  assert.deepEqual(raised(), []);

  // And it is not resurrected by going back.
  route = "/chat";
  dismissStartToasts();
  assert.deepEqual(raised(), []);
});

test("a toast raised on the route it landed on stays", () => {
  // A start can itself navigate, and the toast then belongs where it appeared;
  // dismissing every live toast on any route change would take it straight back out.
  reset("/hub");
  showStartToast("model:repo:q4", MESSAGE);
  calls.length = 0;

  dismissStartToasts();
  assert.deepEqual(raised(), []);
});

test("a noticeOnly caller is folded in or not shown at all", () => {
  // #9663 removed chat's own auto-load toast as a duplicate of the download panel.
  // Its sentence still rides along for the Xet notice to fold in, but on an HTTP
  // start, or once the three notices are spent, it must not come back on its own.
  reset();
  showCallerToast("model:repo:q4", {
    title: "Downloading model",
    description: "It'll load automatically once the download finishes.",
    noticeOnly: true,
  });
  assert.deepEqual(raised(), []);

  showCallerToast("model:repo:q4", {
    title: "Downloading in the background",
    description: "It'll be ready to load once the current model finishes.",
  });
  assert.equal(calls.length, 1);
  assert.equal(calls[0].title, "Downloading in the background");
});

test("a caller line that has gone stale is dropped", () => {
  // Chat moved to another thread while the start was still in flight, so nothing
  // auto-loads any more. The transfer is still running, so the notice it would have
  // been folded into stays true; only the caller's promise goes.
  reset();
  let context = "thread-a";
  const caller = {
    title: "Downloading model",
    description: "It'll load automatically once the download finishes.",
    stillValid: () => context === "thread-a",
  };
  assert.equal(liveCallerToast(caller), caller);

  context = "thread-b";
  assert.equal(liveCallerToast(caller), undefined);

  // No predicate means always valid, which is every other surface.
  assert.equal(liveCallerToast(MESSAGE), MESSAGE);
  assert.equal(liveCallerToast(undefined), undefined);
});

test("a caller with nothing to say is silent", () => {
  reset();
  showCallerToast("model:repo:q4", undefined);
  assert.deepEqual(raised(), []);
});
