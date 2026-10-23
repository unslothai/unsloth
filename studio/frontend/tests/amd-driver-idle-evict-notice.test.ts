// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The System tab's notice for a Windows AMD driver with the idle-eviction bug
// (ROCm/TheRock#7221), from /api/system's gpu.driver_warning. Lifted by regex like
// gpu-torch-mismatch.test.ts, since the tab pulls in the whole app.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrcAsync, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { en } = await import("../src/i18n/locales/en.ts");
const tabSrc = await readSrcAsync("features/settings/tabs/resources-tab.tsx");

function lift(pattern: RegExp, what: string): string {
  const found = pattern.exec(tabSrc);
  assert.ok(found, `could not find ${what} in resources-tab.tsx`);
  return found[0];
}

const derivation = [
  lift(/const rawDriverWarning = [\s\S]*?;/, "rawDriverWarning"),
  lift(/const driverWarning =[\s\S]*?;/, "driverWarning"),
].join("\n");

interface Warning {
  driver_version?: string;
  severity?: string;
  link?: string;
}

function noticeFor(
  gpuInventory: { driver_warning?: Warning | null } | null,
  dismissedDriverWarning: string | null = null,
) {
  const run = new Function(
    "gpuInventory",
    "dismissedDriverWarning",
    `${derivation}\n return driverWarning;`,
  );
  return run(gpuInventory, dismissedDriverWarning) as Warning | null;
}

const BROKEN = {
  driver_version: "32.0.31041.1004",
  severity: "warning",
  link: "https://github.com/ROCm/TheRock/issues/7221",
};

test("a flagged driver shows the notice", () => {
  assert.deepEqual(noticeFor({ driver_warning: BROKEN }), BROKEN);
});

test("no warning, or a host not read yet, shows nothing", () => {
  for (const inventory of [
    null,
    {},
    { driver_warning: null },
    { driver_warning: {} },
  ]) {
    assert.equal(noticeFor(inventory), null);
  }
});

test("dismissing hides that driver only, so a different bad driver shows again", () => {
  assert.equal(noticeFor({ driver_warning: BROKEN }, "32.0.31041.1004"), null);
  const other = { ...BROKEN, driver_version: "32.0.31014.1001" };
  assert.deepEqual(
    noticeFor({ driver_warning: other }, "32.0.31041.1004"),
    other,
  );
});

test("the message names the version and the fixed release", () => {
  const message = en.settings.resources.gpu.driverIdleEvict;
  assert.match(message, /\{version\}/);
  assert.match(message, /Adrenalin 26\.9\.2/);
});

test("only an https link is rendered", () => {
  assert.match(tabSrc, /driverWarning\.link\?\.startsWith\("https:\/\/"\)/);
});
