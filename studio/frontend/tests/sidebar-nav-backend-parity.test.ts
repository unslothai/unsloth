// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { DEFAULT_CUSTOMIZATION } from "../src/features/settings/stores/appearance-custom-store.ts";

import { readSrcAsync, readText } from "./helpers/kit.ts";

// Old records get the backend's defaults, so they must match; read the real constant.
test("the backend sidebar nav defaults match the frontend", async () => {
  const source = readText("../../backend/routes/settings.py");
  const block = /SIDEBAR_NAV_ITEM_DEFAULTS = \{([\s\S]*?)^\}/m.exec(source);
  assert.ok(block, "could not find SIDEBAR_NAV_ITEM_DEFAULTS in settings.py");
  const backend = [...block[1].matchAll(/"([a-z]+)":\s*(True|False)/g)].map((m) => ({
    id: m[1],
    pinned: m[2] === "True",
  }));
  // Order matters: the backend appends missing ids in this order.
  assert.deepEqual(backend, DEFAULT_CUSTOMIZATION.sidebarNav);
});

// PersonalizationCustomization ignores unknown keys, so undeclared fields are lost on save.
test("the backend stores every customization field the frontend sends", () => {
  const source = readText("../../backend/routes/settings.py");
  const model = /class PersonalizationCustomization\(BaseModel\):([\s\S]*?)\nclass /.exec(
    source,
  );
  assert.ok(model, "could not find PersonalizationCustomization in settings.py");
  const fields = new Set(
    [...model[1].matchAll(/^ {4}(\w+):/gm)]
      .map((m) => m[1])
      .filter((name) => name !== "model_config"),
  );
  for (const key of Object.keys(DEFAULT_CUSTOMIZATION)) {
    assert.ok(fields.has(key), `the backend drops "${key}" on every save`);
  }
  // Defaults to None so a pre-field record differs from a user-chosen empty list.
  assert.match(
    model[1],
    /sidebarNavAuto: Optional\[list\[SidebarNavItemId\]\] = Field\(\n\s*None,/,
  );
});

// navRows is keyed by id, so a rename silently un-gates a capability row.
test("Train and Video are still the capability-gated rows", async () => {
  const source = await readSrcAsync("components/app-sidebar.tsx");
  const rows = /const navRows: Record<SidebarNavItemId, NavRowDef> = \{([\s\S]*?)\n  \};/.exec(
    source,
  );
  assert.ok(rows, "could not find navRows in app-sidebar.tsx");
  const bodies = new Map<string, string>();
  const keys = [...rows[1].matchAll(/^    ([a-z]+): \{$/gm)];
  keys.forEach((key, i) => {
    const start = key.index + key[0].length;
    const end = i + 1 < keys.length ? keys[i + 1].index : rows[1].length;
    bodies.set(key[1], rows[1].slice(start, end));
  });
  const backend = readText("../../backend/routes/settings.py");
  const block = /SIDEBAR_NAV_ITEM_DEFAULTS = \{([\s\S]*?)^\}/m.exec(backend);
  assert.ok(block, "could not find SIDEBAR_NAV_ITEM_DEFAULTS in settings.py");
  for (const [, id] of block[1].matchAll(/"([a-z]+)":/g)) {
    assert.ok(bodies.has(id), `the backend ships a "${id}" row the sidebar does not define`);
  }

  // Video's disabled expression is pinned in provisional-hardware-verdict.test.ts.
  for (const [id, disabled] of [
    ["train", /disabled: chatOnlyMeasured,/],
    ["video", /disabled: (?!chatOnlyMeasured)\w+,/],
  ] as const) {
    const body = bodies.get(id);
    assert.ok(body, `no ${id} row`);
    assert.match(
      body,
      /pending: capabilitiesUnknown,/,
      `the ${id} row renders its disabled state before the verdict is measured`,
    );
    assert.match(
      body,
      disabled,
      `the ${id} row is no longer capability-gated on its own verdict`,
    );
  }
});
