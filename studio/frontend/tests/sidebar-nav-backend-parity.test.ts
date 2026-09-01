// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { DEFAULT_CUSTOMIZATION } from "../src/features/settings/stores/appearance-custom-store.ts";

// A record predating sidebarNav is served the backend's own defaults, so a drift there
// hands the user a layout this side never shipped. settings.py says the two must match;
// the backend's parity test compares against a hand-copied list, which cannot catch a
// frontend-only change. Read the real constant instead.
test("the backend sidebar nav defaults match the frontend", async () => {
  const source = await readFile(
    new URL("../../backend/routes/settings.py", import.meta.url),
    "utf8",
  );
  const block = /SIDEBAR_NAV_ITEM_DEFAULTS = \{([\s\S]*?)^\}/m.exec(source);
  assert.ok(block, "could not find SIDEBAR_NAV_ITEM_DEFAULTS in settings.py");
  const backend = [...block[1].matchAll(/"([a-z]+)":\s*(True|False)/g)].map((m) => ({
    id: m[1],
    pinned: m[2] === "True",
  }));
  // Order matters too: the backend appends its missing ids in this order.
  assert.deepEqual(backend, DEFAULT_CUSTOMIZATION.sidebarNav);
});

// The two capability-gated rows. They are the ones that can render disabled without the user
// having done anything, so they are also the ones a rename would silently un-gate: navRows is
// keyed by SidebarNavItemId, so a dropped `pending` there just stops spinning, it does not
// fail to compile.
test("Train and Video are still the capability-gated rows", async () => {
  const source = await readFile(
    new URL("../src/components/app-sidebar.tsx", import.meta.url),
    "utf8",
  );
  const rows = /const navRows: Record<SidebarNavItemId, NavRowDef> = \{([\s\S]*?)\n  \};/.exec(
    source,
  );
  assert.ok(rows, "could not find navRows in app-sidebar.tsx");
  // Split on the top-level row keys so each row's body can be checked on its own.
  const bodies = new Map<string, string>();
  const keys = [...rows[1].matchAll(/^    ([a-z]+): \{$/gm)];
  keys.forEach((key, i) => {
    const start = key.index + key[0].length;
    const end = i + 1 < keys.length ? keys[i + 1].index : rows[1].length;
    bodies.set(key[1], rows[1].slice(start, end));
  });
  // Every id the backend knows about has a row, or the personalization round-trip renders a gap.
  const backend = await readFile(
    new URL("../../backend/routes/settings.py", import.meta.url),
    "utf8",
  );
  const block = /SIDEBAR_NAV_ITEM_DEFAULTS = \{([\s\S]*?)^\}/m.exec(backend);
  assert.ok(block, "could not find SIDEBAR_NAV_ITEM_DEFAULTS in settings.py");
  for (const [, id] of block[1].matchAll(/"([a-z]+)":/g)) {
    assert.ok(bodies.has(id), `the backend ships a "${id}" row the sidebar does not define`);
  }

  // Train reads the chat-only verdict; Video reads only the subset of its reasons that leave no
  // video device, and that expression is pinned in provisional-hardware-verdict.test.ts, so it is
  // not restated here. Video is still required to have a disabled state -- swapping it for
  // Train's would pass there, and un-gate the row on a Mac whose only problem is MLX.
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
