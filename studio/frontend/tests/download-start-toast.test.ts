// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
  currentStartToastSelectionEpoch,
  dismissStartToastsForModelSelection,
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

// Compare a copy: assert narrows `calls` to never[] and breaks later typechecks.
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
  reset("/chat");
  showStartToast("model:repo:q4", MESSAGE);
  calls.length = 0;

  route = "/hub";
  dismissStartToasts();
  assert.deepEqual(raised(), [
    { kind: "dismiss", id: startToastId("model:repo:q4") },
  ]);

  calls.length = 0;
  route = "/settings";
  dismissStartToasts();
  assert.deepEqual(raised(), []);
});

test("a raise that lands after the user left is dropped", () => {
  reset("/chat");
  const startedOn = "/chat";
  route = "/hub";
  showStartToast("model:repo:q4", MESSAGE, startedOn);
  assert.deepEqual(raised(), []);

  route = "/chat";
  dismissStartToasts();
  assert.deepEqual(raised(), []);
});

test("a toast raised on the route it landed on stays", () => {
  reset("/hub");
  showStartToast("model:repo:q4", MESSAGE);
  calls.length = 0;

  dismissStartToasts();
  assert.deepEqual(raised(), []);
});

test("a later model selection drops start toasts without a route change", () => {
  reset("/chat");
  showStartToast("model:old-repo:q4", {
    title: "Restarting this download",
    description: "The partial can't be resumed.",
  });
  calls.length = 0;

  dismissStartToastsForModelSelection();
  assert.deepEqual(raised(), [
    { kind: "dismiss", id: startToastId("model:old-repo:q4") },
  ]);
});

test("a model pick preserves dataset notices and their delayed raises", () => {
  reset("/hub");
  const modelKey = "model:model-repo:q4";
  const datasetKey = "dataset:dataset-repo:main";
  showStartToast(modelKey, MESSAGE);
  showStartToast(datasetKey, MESSAGE);
  const startedForSelection = currentStartToastSelectionEpoch();
  calls.length = 0;

  dismissStartToastsForModelSelection();
  assert.deepEqual(raised(), [{ kind: "dismiss", id: startToastId(modelKey) }]);

  calls.length = 0;
  showStartToast(datasetKey, MESSAGE, "/hub", startedForSelection);
  assert.equal(calls[0]?.kind, "info");
  assert.equal(calls[0]?.options?.id, startToastId(datasetKey));
  dismissStartToast(datasetKey);
});
test("a delayed start cannot reappear after the next model selection", () => {
  reset("/chat");
  const startedForSelection = currentStartToastSelectionEpoch();

  dismissStartToastsForModelSelection();
  calls.length = 0;
  showStartToast("model:old-repo:q4", MESSAGE, "/chat", startedForSelection);

  assert.deepEqual(raised(), []);
});

test("a noticeOnly caller is folded in or not shown at all", () => {
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

  assert.equal(liveCallerToast(MESSAGE), MESSAGE);
  assert.equal(liveCallerToast(undefined), undefined);
});

test("a caller with nothing to say is silent", () => {
  reset();
  showCallerToast("model:repo:q4", undefined);
  assert.deepEqual(raised(), []);
});
