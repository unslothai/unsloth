// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import {
  AppPortalGate,
  AppRevealedContext,
  createAppReadinessScope,
} from "../src/components/app-readiness.ts";
import { readSrc } from "./helpers/kit.ts";

test("late completion cannot reveal either an unavailable app or its new generation", async () => {
  const target = new EventTarget();
  const previousWindow = Object.getOwnPropertyDescriptor(globalThis, "window");
  Object.defineProperty(globalThis, "window", { configurable: true, value: target });
  try {
    let reveals = 0;
    let reloadSignals = 0;
    target.addEventListener("unsloth:app-shell-ready", () => reloadSignals++);
    const oldMount = createAppReadinessScope(() => reveals++);
    // Capture the callback before awaiting, just like the history adapter.
    const oldCompleteLoad = oldMount.signalReady;
    oldMount.dispose();
    await Promise.resolve().then(oldCompleteLoad);
    assert.equal(reveals, 0, "must ignore completion while backend is unavailable");
    const newMount = createAppReadinessScope(() => reveals++);
    await Promise.resolve().then(oldCompleteLoad);
    assert.equal(reveals, 0, "must also ignore old work after the replacement mounts");
    assert.equal(reloadSignals, 0, "stale completion must not retire a reload snapshot either");
    newMount.signalReady();
    assert.equal(reveals, 1);
    assert.equal(reloadSignals, 1, "current readiness still notifies reload-snapshot.js");
    newMount.dispose();
    newMount.signalReady();
    assert.equal(reveals, 1);
  } finally {
    if (previousWindow) Object.defineProperty(globalThis, "window", previousWindow);
    else Reflect.deleteProperty(globalThis, "window");
  }
});

test("retained-open portal children do not mount until reveal, without changing owner state", () => {
  const owner = { open: true, tab: "api-keys" };
  let childMounts = 0;
  function PortalChild() {
    childMounts++;
    return createElement("div", { role: "dialog" }, owner.tab);
  }
  const render = (revealed: boolean) => renderToStaticMarkup(
    createElement(AppRevealedContext.Provider, { value: revealed },
      createElement(AppPortalGate, { children: owner.open ? createElement(PortalChild) : null })),
  );
  assert.equal(render(false), "");
  assert.equal(childMounts, 0);
  assert.deepEqual(owner, { open: true, tab: "api-keys" });
  assert.match(render(true), /role="dialog"/);
  assert.equal(childMounts, 1);
  assert.equal(render(false), "", "backend restart suppresses the retained dialog again");
  assert.equal(childMounts, 1);
  assert.match(render(true), /api-keys/);
  assert.equal(childMounts, 2);
  for (const file of ["dialog", "sheet", "alert-dialog"]) {
    assert.match(readSrc(`components/ui/${file}.tsx`), /<AppPortalGate>[\s\S]*?Primitive\.Portal[\s\S]*?<\/AppPortalGate>/);
  }
});

test("web portals retain their default visible behavior", () => {
  assert.equal(renderToStaticMarkup(createElement(AppPortalGate, {
    children: createElement("span", null, "visible"),
  })), "<span>visible</span>");
});
