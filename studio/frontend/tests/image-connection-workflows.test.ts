// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { CONNECTION_WORKFLOWS, WORKFLOW_TABS } from "../src/features/images/workflows.ts";
import { readSrc } from "./helpers/kit.ts";

const PAGE = readSrc("features/images/images-page.tsx");

test("a connection offers Create and Edit, and no other image workflow", () => {
  assert.deepEqual([...CONNECTION_WORKFLOWS], ["create", "edit"]);
  const off = WORKFLOW_TABS.map((tab) => tab.id).filter((id) => !CONNECTION_WORKFLOWS.includes(id));
  assert.deepEqual(off, ["transform", "inpaint", "extend", "upscale", "reference"]);
});

// The page is one component, so its wiring is read from source.
test("the Images page publishes that list while a connection is selected", () => {
  assert.match(PAGE, /if \(external\) \{\s+setSupported\(\[\.\.\.CONNECTION_WORKFLOWS\]\);\s+return;/);
  assert.match(PAGE, /!unifiedEditActive && !\(external && workflow === "edit"\) && sizeControls/);
});
