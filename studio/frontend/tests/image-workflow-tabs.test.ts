// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { imageWorkflowTabs, WORKFLOW_TABS } from "../src/features/images/workflows.ts";

test("keeps separate Edit and Reference entries for non-unified models", () => {
  assert.deepEqual(imageWorkflowTabs(false), WORKFLOW_TABS);
  assert.deepEqual(
    imageWorkflowTabs(false).map((tab) => tab.id),
    ["create", "edit", "transform", "inpaint", "extend", "upscale", "reference"],
  );
});

test("collapses unified Edit and Reference into one backwards-compatible entry", () => {
  const tabs = imageWorkflowTabs(true);
  assert.deepEqual(
    tabs.map((tab) => tab.id),
    ["create", "edit", "transform", "inpaint", "extend", "upscale"],
  );
  const edit = tabs.find((tab) => tab.id === "edit");
  assert.equal(edit?.label, "Edit / Reference");
  assert.match(edit?.hint ?? "", /reference/i);
});
