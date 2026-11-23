// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// AnimatePresence keeps an exiting panel mounted, so its cleanup runs after the replacement.

// The store's half is asserted directly; the panel's half by reading the
// source, since the node suite has no DOM to mount two panels into.

import assert from "node:assert/strict";
import test from "node:test";

import {
  type MonitorFrame,
  useMonitorFrameStore,
} from "../src/features/settings/stores/monitor-frame-store.ts";

import { readSrc } from "./helpers/kit.ts";

const PANEL_SOURCE = readSrc("hooks/use-floating-panel-layout.ts");

const ROOT_SOURCE = readSrc("app/routes/__root.tsx");
const SETTINGS_MOUNT_SOURCE = readSrc(
  "features/settings/settings-dialog-mount.tsx",
);

function corner(height = 300): MonitorFrame {
  return { left: 1168, top: 884 - height, right: 1424, bottom: 884 };
}

function published(): MonitorFrame[] {
  return [...useMonitorFrameStore.getState().frames.values()];
}

function reset(): void {
  useMonitorFrameStore.setState({ frames: new Map() });
}

test("a panel publishes its own box", () => {
  reset();
  const panel = {};
  useMonitorFrameStore.getState().setFrame(panel, corner());
  assert.deepEqual(published(), [corner()]);
});

test("closing the only monitor clears the frame", () => {
  reset();
  const panel = {};
  useMonitorFrameStore.getState().setFrame(panel, corner());
  useMonitorFrameStore.getState().clearFrame(panel);
  assert.deepEqual(published(), []);
});

test("an exiting panel does not clear the replacement's frame", () => {
  reset();
  const closing = {};
  const reopened = {};
  useMonitorFrameStore.getState().setFrame(closing, corner(300));
  useMonitorFrameStore.getState().setFrame(reopened, corner(220));

  useMonitorFrameStore.getState().clearFrame(closing);

  assert.deepEqual(
    published(),
    [corner(220)],
    "the open monitor's frame must survive the old panel's unmount",
  );
  // A still monitor republishes nothing, so a lost frame stays lost.
  assert.deepEqual(
    [...useMonitorFrameStore.getState().frames.keys()],
    [reopened],
    "only the panel that is still open may still be published",
  );
});

test("the replacement can still clear its own frame when closed", () => {
  reset();
  const closing = {};
  const reopened = {};
  useMonitorFrameStore.getState().setFrame(closing, corner(300));
  useMonitorFrameStore.getState().setFrame(reopened, corner(220));
  useMonitorFrameStore.getState().clearFrame(closing);
  useMonitorFrameStore.getState().clearFrame(reopened);
  assert.deepEqual(published(), []);
});

// The stack re-renders per notification and ResizeObserver fires often, so skip no-ops.
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
  assert.deepEqual(published(), [corner()]);
});

// The store keeps every published box, so the stack dodges the card and composer together.
test("two publishers are dodged together, not one at a time", () => {
  reset();
  const monitor = {};
  const composer = {};
  useMonitorFrameStore.getState().setFrame(monitor, corner(300));
  useMonitorFrameStore
    .getState()
    .setFrame(composer, { left: 300, top: 780, right: 1100, bottom: 860 });
  assert.deepEqual(
    published(),
    [corner(300), { left: 300, top: 780, right: 1100, bottom: 860 }],
    "both are kept, apart, for panel-placement to dodge one at a time",
  );
});

test("dropping one publisher leaves the other's box intact", () => {
  reset();
  const monitor = {};
  const composer = {};
  const composerBox = { left: 300, top: 780, right: 1100, bottom: 860 };
  useMonitorFrameStore.getState().setFrame(monitor, corner(300));
  useMonitorFrameStore.getState().setFrame(composer, composerBox);
  useMonitorFrameStore.getState().clearFrame(monitor);
  assert.deepEqual(published(), [composerBox]);
});

// A hidden composer measures 0x0, which would pin the stack to the top-left.
test("the publish hook drops an unmeasurable box rather than publishing it", () => {
  const HOOK = readSrc("features/settings/hooks/use-published-frame.ts");
  assert.match(HOOK, /box\.width === 0 && box\.height === 0/);
  assert.match(HOOK, /observer\?\.disconnect\(\)/, "and it must unsubscribe");
  assert.match(
    HOOK,
    /clearFrame\(publisher\);\s*\n\s*\};/,
    "and clear on unmount",
  );
});

test("settings and monitor are eagerly imported and mounted without outer loading UI", () => {
  assert.match(SETTINGS_MOUNT_SOURCE, /import \{ SettingsDialog \} from "\.\/settings-dialog"/);
  assert.match(SETTINGS_MOUNT_SOURCE, /import \{ FloatingMonitor \} from "@\/components\/floating-monitor"/);
  assert.match(SETTINGS_MOUNT_SOURCE, /<SettingsDialog \/>/);
  assert.match(SETTINGS_MOUNT_SOURCE, /<FloatingMonitor \/>/);
  assert.doesNotMatch(SETTINGS_MOUNT_SOURCE, /lazy\(|Suspense|LazyImport|settingsMounted|monitorMounted|settingsOpen|monitorOpen|settings-dialog-loading/);
});

test("eager settings surfaces remain gated on auth and credential readiness", () => {
  assert.match(ROOT_SOURCE, /<CredentialBootstrapGate active=\{!isAuthFlowRoute\}>/);
  assert.match(ROOT_SOURCE, /<SettingsDialogMount active=\{active && ready\} \/>/);
  assert.match(SETTINGS_MOUNT_SOURCE, /if \(!active\) return null;/);
});
