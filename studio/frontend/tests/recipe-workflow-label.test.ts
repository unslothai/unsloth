// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";

import { recipeWorkflowLabel, recipeWorkflowTab } from "../src/features/images/workflows.ts";
import { readSrc } from "./helpers/kit.ts";

test("recipeWorkflowLabel maps backend workflow ids to UI labels", () => {
  assert.equal(recipeWorkflowLabel(null), "Create");
  assert.equal(recipeWorkflowLabel("txt2img"), "Create");
  assert.equal(recipeWorkflowLabel("img2img"), "Transform");
  assert.equal(recipeWorkflowLabel("outpaint"), "Extend");
  assert.equal(recipeWorkflowLabel("reference"), "Reference");
});

test("restoring an Extend recipe still asks for its source image", () => {
  const src = readSrc("features/images/images-page.tsx");
  const table = src.slice(src.indexOf("const CONDITIONED_WORKFLOW_INPUTS"));
  assert.match(table.slice(0, table.indexOf("};")), /\boutpaint: "the source image",/);
});

test("recipeWorkflowTab maps backend workflow ids to their tabs", () => {
  const tab = recipeWorkflowTab;
  assert.equal(tab(null), "create");
  assert.equal(tab("txt2img"), "create");
  assert.equal(tab("controlnet"), "create");
  assert.equal(tab("img2img"), "transform");
  assert.equal(tab("inpaint"), "inpaint");
  assert.equal(tab("outpaint"), "extend");
  assert.equal(tab("upscale"), "upscale");
  assert.equal(tab("edit"), "edit");
  assert.equal(tab("reference"), "reference");
});
