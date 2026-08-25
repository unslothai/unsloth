// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
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
  [
    { id: "hub", pinned: true },
    { id: "projects", pinned: true },
    { id: "images", pinned: true },
    { id: "video", pinned: false },
    { id: "audio", pinned: false },
    { id: "train", pinned: true },
    { id: "recipes", pinned: false },
    { id: "export", pinned: false },
    { id: "api", pinned: false },
  ],
  [
    { id: "hub", pinned: true },
    { id: "projects", pinned: true },
    { id: "images", pinned: true },
    { id: "video", pinned: true },
    { id: "audio", pinned: false },
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
      migrateShippedSidebarNavDefault(customization, 4, 8).sidebarNav,
      DEFAULT_CUSTOMIZATION.sidebarNav,
    );
  }
});

test("the untouched pre-Audio version-5 sidebar adopts the Audio-aware default", () => {
  const customization = sanitizeCustomization({
    sidebarNav: shippedLayouts[3],
  });
  assert.deepEqual(
    migrateShippedSidebarNavDefault(customization, 5, 8).sidebarNav,
    DEFAULT_CUSTOMIZATION.sidebarNav,
  );
});

test("a customized version-5 sidebar keeps its order and only gains later ids", () => {
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
    migrateShippedSidebarNavDefault(customization, 5, 8),
    customization,
  );
  assert.deepEqual(
    customization.sidebarNav.map((item) => item.id),
    [...customizedV5.map((item) => item.id), "notebooks", "audio"],
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
    migrateShippedSidebarNavDefault(customization, 4, 8),
    customization,
  );
});

test("an install sitting on the version-6 default picks Video up", () => {
  // The layout this change replaces. Untouched, so it adopts the new default
  // rather than being read as a deliberate choice to keep Video under More.
  const customization = sanitizeCustomization({ sidebarNav: shippedLayouts[4] });
  const migrated = migrateShippedSidebarNavDefault(customization, 6, 8);
  assert.deepEqual(migrated.sidebarNav, DEFAULT_CUSTOMIZATION.sidebarNav);
  const ids = migrated.sidebarNav.filter((item) => item.pinned).map((i) => i.id);
  assert.deepEqual(ids, ["hub", "projects", "images", "notebooks", "video", "train"]);
});

test("a shipped-looking layout chosen after migration is preserved", () => {
  const customization = sanitizeCustomization({
    sidebarNav: shippedLayouts[2],
  });
  assert.strictEqual(
    migrateShippedSidebarNavDefault(customization, 8, 8),
    customization,
  );
});

/** The sync module uses path aliases, so read its constant rather than import it. */
async function personalizationVersion(): Promise<number> {
  const source = await readFile(
    new URL(
      "../src/features/profile/hooks/use-personalization-sync.ts",
      import.meta.url,
    ),
    "utf8",
  );
  const match = /PERSONALIZATION_VERSION = (\d+)/.exec(source);
  assert.ok(match, "no PERSONALIZATION_VERSION in the sync module");
  return Number(match[1]);
}

test("a synced profile picks the layout change up too", async () => {
  // Remote hydration replaces the local store wholesale, so a nav default that
  // only migrates locally is overwritten by the stored layout on every login.
  // PERSONALIZATION_VERSION has to move with the layout for that migration to
  // run against the remote record.
  const stored = sanitizeCustomization({ sidebarNav: shippedLayouts[4] });
  const migrated = migrateShippedSidebarNavDefault(
    stored,
    3,
    await personalizationVersion(),
  );
  assert.deepEqual(migrated.sidebarNav, DEFAULT_CUSTOMIZATION.sidebarNav);
});
