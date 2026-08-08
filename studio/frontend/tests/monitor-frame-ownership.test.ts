// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Closing and reopening the Live monitor inside its exit animation leaves two
// panels mounted at once: AnimatePresence keeps the leaving child rendered and
// drops it in a later commit (framer-motion's AnimatePresence calls
// setRenderedChildren(pendingPresentChildren.current) only once every exit has
// completed), so the replacement mounts and publishes first and the old panel
// unmounts last. Its cleanup must not take the replacement's frame with it.

// The store's half is asserted directly; the panel's half by reading the
// source, since the node suite has no DOM to mount two panels into.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  type MonitorFrame,
  useMonitorFrameStore,
} from "../src/features/settings/stores/monitor-frame-store.ts";

const PANEL_SOURCE = readFileSync(
  fileURLToPath(
    new URL("../src/components/floating-monitor.tsx", import.meta.url),
  ),
  "utf8",
);

/** The Live monitor where it opens by default: bottom-right, w-64, inset-4. */
function corner(height = 300): MonitorFrame {
  return { left: 1168, top: 884 - height, right: 1424, bottom: 884 };
}

function reset(): void {
  useMonitorFrameStore.setState({ frame: null, publisher: null });
}

test("a panel publishes its own box", () => {
  reset();
  const panel = {};
  useMonitorFrameStore.getState().setFrame(panel, corner());
  assert.deepEqual(useMonitorFrameStore.getState().frame, corner());
});

test("closing the only monitor clears the frame", () => {
  reset();
  const panel = {};
  useMonitorFrameStore.getState().setFrame(panel, corner());
  useMonitorFrameStore.getState().clearFrame(panel);
  assert.equal(useMonitorFrameStore.getState().frame, null);
});

// The regression: reopened during the exit animation, so the replacement
// publishes before the panel it replaced is torn down.
test("an exiting panel does not clear the replacement's frame", () => {
  reset();
  const closing = {};
  const reopened = {};
  useMonitorFrameStore.getState().setFrame(closing, corner(300));
  useMonitorFrameStore.getState().setFrame(reopened, corner(220));

  useMonitorFrameStore.getState().clearFrame(closing);

  assert.deepEqual(
    useMonitorFrameStore.getState().frame,
    corner(220),
    "the open monitor's frame must survive the old panel's unmount",
  );
  // A monitor that then sits still resizes nothing and republishes nothing, so
  // a lost frame would stay lost and the stack would sit back on top of it.
  assert.equal(useMonitorFrameStore.getState().publisher, reopened);
});

test("the replacement can still clear its own frame when closed", () => {
  reset();
  const closing = {};
  const reopened = {};
  useMonitorFrameStore.getState().setFrame(closing, corner(300));
  useMonitorFrameStore.getState().setFrame(reopened, corner(220));
  useMonitorFrameStore.getState().clearFrame(closing);
  useMonitorFrameStore.getState().clearFrame(reopened);
  assert.equal(useMonitorFrameStore.getState().frame, null);
});

// The overlay stack re-renders on every notification, and reconcileGeometry
// runs on each ResizeObserver delivery, so an unchanged box must not notify.
test("republishing the same box from the same panel does not notify", () => {
  reset();
  const panel = {};
  let notifications = 0;
  const unsubscribe = useMonitorFrameStore.subscribe(() => {
    notifications += 1;
  });
  useMonitorFrameStore.getState().setFrame(panel, corner());
  useMonitorFrameStore.getState().setFrame(panel, corner());
  useMonitorFrameStore.getState().setFrame(panel, corner());
  unsubscribe();
  assert.equal(notifications, 1);
});

// This is what regressed: the unmount cleanup nulled the shared frame outright.
test("the panel's unmount cleanup goes through clearFrame", () => {
  assert.match(
    PANEL_SOURCE,
    /clearFrame\(publisher\)/,
    "the teardown must release only this panel's claim",
  );
  assert.doesNotMatch(
    PANEL_SOURCE,
    /setFrame\(\s*null\s*\)/,
    "no unconditional clear of the shared frame",
  );
  // Every publish carries the owner, so a frame can never be left unowned.
  assert.equal(
    PANEL_SOURCE.match(/setFrame\(publisher,/g)?.length,
    2,
    "both the reconcile and the drag republish name their panel",
  );
});

test("clearing on behalf of a panel that owns nothing does not notify", () => {
  reset();
  const panel = {};
  useMonitorFrameStore.getState().setFrame(panel, corner());
  let notifications = 0;
  const unsubscribe = useMonitorFrameStore.subscribe(() => {
    notifications += 1;
  });
  useMonitorFrameStore.getState().clearFrame({});
  unsubscribe();
  assert.equal(notifications, 0);
  assert.deepEqual(useMonitorFrameStore.getState().frame, corner());
});
