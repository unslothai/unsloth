// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

test("a file dropped on a message reaches the page-wide drop handlers through the portal host", () => {
  const thread = readSrc("components/assistant-ui/thread.tsx");
  const hostStart = thread.indexOf("<ThreadPrimitive.Viewport");
  const portalStart = thread.indexOf("createPortal(", hostStart);
  const rootStart = thread.indexOf("<ThreadPrimitive.Root", portalStart);
  assert.ok(hostStart >= 0 && portalStart > hostStart && rootStart > portalStart);
  const host = thread.slice(hostStart, portalStart);
  const portal = thread.slice(portalStart, rootStart);
  const root = thread.slice(rootStart, thread.indexOf("<IntentAwareScrollProvider", rootStart));
  assert.match(host, /className="aui-thread-host hidden"/);
  assert.ok(portal.includes("<ProgressiveMessages"), "the rows are portaled from the host, so React routes their events to it and never to the root");
  for (const handler of ["onDragEnter", "onDragOver", "onDragLeave", "onDrop"]) {
    assert.ok(host.includes(`${handler}={${handler}}`), `the portal host forwards ${handler}`);
    assert.ok(root.includes(`${handler}={${handler}}`), `the thread root keeps ${handler} for the viewport padding and the composer dock`);
  }
});
