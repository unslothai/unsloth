// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { stripTypeScriptTypes } from "node:module";
import { resolveHubSection } from "../src/features/model-picker/components/model-selector/hub-section.ts";
import { readSrc } from "./helpers/kit.ts";

const selector = readSrc("features/model-picker/components/model-selector.tsx");
const defaultSectionSource = selector.slice(
  selector.indexOf("function defaultHubSection("),
  selector.indexOf("const HUB_SECTION_TABS"),
);
const defaultSection = new Function(
  "loadLastHubSection",
  "hasDownloadedModels",
  stripTypeScriptTypes(defaultSectionSource) + "\nreturn defaultHubSection;",
);

const savedDevice = {
  section: "downloaded" as const,
  hasExternal: false,
  hasDeviceModels: null,
  sectionChosenThisOpen: false,
};

test("saved Device survives cold caches until scoped inventory settles", () => {
  const initialSection = defaultSection(() => "downloaded", () => false)();
  for (const hasDeviceModels of [null, true, false]) {
    assert.equal(
      resolveHubSection({ ...savedDevice, section: initialSection, hasDeviceModels }),
      hasDeviceModels === false ? "recommended" : "downloaded",
    );
  }
});

test("a saved Device tab falls back when this task's inventory is empty", () => {
  assert.equal(resolveHubSection(savedDevice), "downloaded");
  assert.equal(resolveHubSection({ ...savedDevice, hasDeviceModels: false }), "recommended");
  assert.equal(resolveHubSection({ ...savedDevice, hasDeviceModels: true }), "downloaded");
});

test("users can deliberately browse empty Device folders until the next open", () => {
  const empty = { ...savedDevice, hasDeviceModels: false };
  assert.equal(resolveHubSection({ ...empty, sectionChosenThisOpen: true }), "downloaded");
  assert.equal(resolveHubSection(empty), "recommended");
});

test("empty Device inventory preserves Recommended and valid Connected selections", () => {
  const empty = { ...savedDevice, hasDeviceModels: false };
  assert.equal(resolveHubSection({ ...empty, section: "recommended" }), "recommended");
  assert.equal(resolveHubSection({ ...empty, section: "connected", hasExternal: true }), "connected");
  assert.equal(resolveHubSection({ ...empty, section: "connected" }), "recommended");
});
