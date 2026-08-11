// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  DEFAULT_CUSTOMIZATION,
  type SidebarNavItemPref,
  migrateShippedSidebarNavDefault,
  sanitizeCustomization,
} from "../src/features/settings/stores/appearance-custom-store.ts";

const shippedLayouts: SidebarNavItemPref[][] = [
  [
    { id: "projects", pinned: true },
    { id: "hub", pinned: true },
    { id: "images", pinned: true },
    { id: "train", pinned: true },
    { id: "video", pinned: false },
    { id: "recipes", pinned: false },
    { id: "export", pinned: false },
  ],
  [
    { id: "projects", pinned: true },
    { id: "hub", pinned: true },
    { id: "images", pinned: true },
    { id: "video", pinned: true },
    { id: "train", pinned: true },
    { id: "recipes", pinned: false },
    { id: "export", pinned: false },
  ],
  [
    { id: "hub", pinned: true },
    { id: "projects", pinned: true },
    { id: "images", pinned: true },
    { id: "video", pinned: true },
    { id: "train", pinned: true },
    { id: "recipes", pinned: false },
    { id: "export", pinned: false },
  ],
  [
    { id: "hub", pinned: true },
    { id: "projects", pinned: true },
    { id: "images", pinned: true },
    { id: "video", pinned: false },
    { id: "train", pinned: true },
    { id: "recipes", pinned: false },
    { id: "export", pinned: false },
    { id: "api", pinned: false },
  ],
];

test("every previously shipped sidebar adopts the current default", () => {
  for (const sidebarNav of shippedLayouts) {
    const customization = sanitizeCustomization({ sidebarNav });
    assert.deepEqual(
      migrateShippedSidebarNavDefault(customization, 4, 6).sidebarNav,
      DEFAULT_CUSTOMIZATION.sidebarNav,
    );
  }
});

test("the untouched pre-Audio version-5 sidebar adopts the Audio-aware default", () => {
  const customization = sanitizeCustomization({
    sidebarNav: shippedLayouts[3],
  });
  assert.deepEqual(
    migrateShippedSidebarNavDefault(customization, 5, 6).sidebarNav,
    DEFAULT_CUSTOMIZATION.sidebarNav,
  );
});

test("a customized version-5 sidebar keeps its order and only gains Audio", () => {
  const customizedV5: SidebarNavItemPref[] = [
    { id: "projects", pinned: true },
    { id: "hub", pinned: true },
    { id: "train", pinned: true },
    { id: "images", pinned: true },
    { id: "video", pinned: false },
    { id: "recipes", pinned: true },
    { id: "export", pinned: false },
    { id: "api", pinned: false },
  ];
  const customization = sanitizeCustomization({ sidebarNav: customizedV5 });

  assert.strictEqual(
    migrateShippedSidebarNavDefault(customization, 5, 6),
    customization,
  );
  assert.deepEqual(
    customization.sidebarNav.map((item) => item.id),
    [...customizedV5.map((item) => item.id), "audio"],
  );
  assert.equal(customization.sidebarNav.at(-1)?.pinned, false);
});

test("a user-arranged sidebar survives the migration", () => {
  const customization = sanitizeCustomization({
    sidebarNav: DEFAULT_CUSTOMIZATION.sidebarNav.map((item) =>
      item.id === "recipes" ? { ...item, pinned: true } : item,
    ),
  });
  assert.strictEqual(
    migrateShippedSidebarNavDefault(customization, 4, 6),
    customization,
  );
});

test("a shipped-looking layout chosen after migration is preserved", () => {
  const customization = sanitizeCustomization({
    sidebarNav: shippedLayouts[2],
  });
  assert.strictEqual(
    migrateShippedSidebarNavDefault(customization, 6, 6),
    customization,
  );
});
