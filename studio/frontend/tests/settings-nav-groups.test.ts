// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { SETTINGS_TABS } from "../src/features/settings/stores/settings-dialog-store.ts";
import { readSrc } from "./helpers/kit.ts";

const dialog = readSrc("features/settings/settings-dialog.tsx");
const groups = dialog.slice(
  dialog.indexOf("export const SETTINGS_NAV_GROUPS"),
  dialog.indexOf("\n];", dialog.indexOf("export const SETTINGS_NAV_GROUPS")),
);
const tabsBlock = dialog.slice(
  dialog.indexOf("const TABS: TabDef[]"),
  dialog.indexOf("\n];", dialog.indexOf("const TABS: TabDef[]")),
);

test("every settings tab sits in exactly one rail group", () => {
  const grouped = [...groups.matchAll(/"([a-z-]+)"/g)]
    .map((m) => m[1])
    .filter((id) => (SETTINGS_TABS as readonly string[]).includes(id));
  const tabs = [...tabsBlock.matchAll(/id: "([a-z-]+)"/g)].map((m) => m[1]);
  assert.deepEqual([...grouped].sort(), [...tabs].sort());
  assert.equal(new Set(grouped).size, grouped.length);
});

test("headed groups name themselves once and vanish when empty", () => {
  assert.match(groups, /headingKey: "settings\.navGroups\.connections"/);
  assert.match(groups, /headingKey: "settings\.navGroups\.dataAccess"/);
  // Tabs keep their TABS order inside a group, whatever order the group lists them in.
  assert.match(
    dialog,
    /tabs: visibleTabs\.filter\(\(tab\) => group\.ids\.includes\(tab\.id\)\)/,
  );
  assert.match(dialog, /\.filter\(\(group\) => group\.tabs\.length > 0\)/);
  assert.match(dialog, /role="group"\s+aria-labelledby=\{headingId\}/);
  // Stacked, the rail is one row: no headings, groups flattened into it.
  assert.match(
    dialog,
    /group-data-stacked\/settings:hidden"\s*>\s*\{t\(group\.headingKey\)\}/,
  );
});
