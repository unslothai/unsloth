// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// #11245: the model load toast had a corner close button that only hid it, next to Cancel, so it
// read as two cancel buttons. The Chat, Images and Video load toasts share labelled controls instead.

import assert from "node:assert/strict";
import test from "node:test";

import {
  MODEL_LOAD_TOAST_CLASSNAMES,
  modelLoadToastControls,
} from "../src/features/chat/lib/model-load-toast.ts";
import { readText } from "./helpers/kit.ts";

test("load toast offers Cancel loading and Hide, and no close button", () => {
  let cancelled = 0;
  let hidden = 0;
  const controls = modelLoadToastControls(
    () => (cancelled += 1),
    () => (hidden += 1),
  );
  assert.equal(controls.closeButton, false);
  assert.equal(controls.cancel?.label, "Cancel loading");
  assert.equal(controls.action.label, "Hide");
  assert.equal(controls.classNames, MODEL_LOAD_TOAST_CLASSNAMES);
  controls.action.onClick();
  assert.deepEqual([cancelled, hidden], [0, 1], "Hide must not cancel the load");
  controls.cancel?.onClick();
  assert.deepEqual([cancelled, hidden], [1, 1]);
});

test("load toast without a cancel handler still offers Hide", () => {
  const controls = modelLoadToastControls(undefined, () => {});
  assert.equal("cancel" in controls, false);
  assert.equal(controls.action.label, "Hide");
});

test("chat load toast uses the shared controls", () => {
  const SOURCE = readText("../src/features/chat/hooks/use-chat-model-runtime.ts");
  assert.match(SOURCE, /\.\.\.modelLoadToastControls\(cancelLoading, /);
  assert.doesNotMatch(SOURCE, /label: "Cancel",\s*onClick: cancelLoading/);
});

test("a cancelled load's progress poll cannot repaint the toast after a replacement starts", () => {
  // The replacement load sets loadingModelRef again, so a poll that resolves late must check its
  // own attempt's abort signal after the await, not only loadingModelRef.
  const SOURCE = readText("../src/features/chat/hooks/use-chat-model-runtime.ts");
  for (const awaited of [
    "getDownloadProgress(progressModelId, hfToken, mlxLoadProgress),",
    "const prog = await getLoadProgress();",
  ]) {
    const at = SOURCE.indexOf(awaited);
    assert.ok(at >= 0, `expected the awaited poll: ${awaited}`);
    const guard = SOURCE.slice(at).match(/\n\s*(if \([^\n]*\) return;)/)?.[1] ?? "";
    assert.match(guard, /abortCtrl\.signal\.aborted/, `stale guard after ${awaited}: ${guard}`);
  }
});
