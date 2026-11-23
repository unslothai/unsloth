// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Panel mounts in a stable host: under Reduce motion in Glass, a direct child of the root restyled
// the whole app root on each open and close (about 1,700 vs 50 elements)

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const CONDITIONAL_HOST = /visible\s*&&\s*\(\s*<div data-api-monitor-host/;
const LAYOUT_FREE_HOST = /data-api-monitor-host=""\s+className="contents"/;
const ROOT_MOUNT = /\{!isAuthFlowRoute && <ApiMonitorOverlay \/>\}/;

const overlay = readSrc("features/api-monitor/api-monitor-overlay.tsx");
const root = readSrc("app/routes/__root.tsx");

function overlayReturn(): string {
  const start = overlay.indexOf("export function ApiMonitorOverlay");
  assert.ok(start >= 0, "ApiMonitorOverlay not found");
  const body = overlay.slice(
    start,
    overlay.indexOf("\nfunction ApiMonitorPanel", start),
  );
  const ret = body.lastIndexOf("return (");
  assert.ok(ret >= 0, "ApiMonitorOverlay has no JSX return");
  return body.slice(ret);
}

test("the API monitor panel mounts inside a host that stays mounted", () => {
  const jsx = overlayReturn();
  const host = jsx.indexOf("<div data-api-monitor-host");
  const presence = jsx.indexOf("<AnimatePresence");
  assert.ok(
    host >= 0,
    "ApiMonitorOverlay must return its always-mounted host element",
  );
  assert.ok(
    presence > host,
    "AnimatePresence (and the panel) must sit inside the host",
  );
  assert.ok(
    !CONDITIONAL_HOST.test(jsx),
    "the host itself must not be conditional",
  );
  assert.match(jsx, LAYOUT_FREE_HOST, "the host must not take part in layout");
});

test("the overlay is still rendered where the root layout always had it", () => {
  assert.match(root, ROOT_MOUNT);
});
