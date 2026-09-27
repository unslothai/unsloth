// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";

import { recipeWorkflowLabel } from "../src/features/images/workflows.ts";

test("recipeWorkflowLabel maps backend workflow ids to UI labels", () => {
  assert.equal(recipeWorkflowLabel(null), "Create");
  assert.equal(recipeWorkflowLabel("txt2img"), "Create");
  assert.equal(recipeWorkflowLabel("img2img"), "Transform");
  assert.equal(recipeWorkflowLabel("outpaint"), "Extend");
  assert.equal(recipeWorkflowLabel("reference"), "Reference");
});
