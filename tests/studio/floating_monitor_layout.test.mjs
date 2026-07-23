// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { getFloatingMonitorLayout } from "../../studio/frontend/src/components/floating-monitor-layout.ts";

test("closed monitor stays hidden", () => {
  assert.deepEqual(
    getFloatingMonitorLayout({
      isOpen: false,
      isMobile: false,
      isChatRoute: true,
      settingsPanelOpen: true,
    }),
    { visible: false, dockedBesideRunSettings: false },
  );
});

test("desktop monitor docks beside open run settings", () => {
  assert.deepEqual(
    getFloatingMonitorLayout({
      isOpen: true,
      isMobile: false,
      isChatRoute: true,
      settingsPanelOpen: true,
    }),
    { visible: true, dockedBesideRunSettings: true },
  );
});

test("mobile monitor yields to the run-settings sheet", () => {
  assert.deepEqual(
    getFloatingMonitorLayout({
      isOpen: true,
      isMobile: true,
      isChatRoute: true,
      settingsPanelOpen: true,
    }),
    { visible: false, dockedBesideRunSettings: false },
  );
});

test("monitor remains visible when run settings are closed", () => {
  assert.deepEqual(
    getFloatingMonitorLayout({
      isOpen: true,
      isMobile: true,
      isChatRoute: true,
      settingsPanelOpen: false,
    }),
    { visible: true, dockedBesideRunSettings: false },
  );
});

test("stale chat settings state does not dock the monitor off-route", () => {
  assert.deepEqual(
    getFloatingMonitorLayout({
      isOpen: true,
      isMobile: false,
      isChatRoute: false,
      settingsPanelOpen: true,
    }),
    { visible: true, dockedBesideRunSettings: false },
  );
});
