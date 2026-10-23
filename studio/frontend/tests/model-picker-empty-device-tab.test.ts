// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { resolveHubSection } from "../src/features/model-picker/components/model-selector/hub-section.ts";

const savedDevice = {
  section: "downloaded" as const,
  hasExternal: false,
  hasDeviceModels: null,
  sectionChosenThisOpen: false,
};

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
