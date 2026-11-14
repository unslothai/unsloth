// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { resolveExportTab } from "../src/features/export/constants.ts";

import { readSrcAsync } from "./helpers/kit.ts";

test("the Q4NX tab falls back to LLM without an NPU", () => {
  assert.equal(resolveExportTab("q4nx", false), "llm");
  assert.equal(resolveExportTab("q4nx", true), "q4nx");
  assert.equal(resolveExportTab("diffusion-lora", false), "diffusion-lora");
  assert.equal(resolveExportTab("llm", true), "llm");
});

test("the Export page puts each flow behind its own tab", async () => {
  const page = await readSrcAsync("features/export/export-page.tsx");
  for (const tab of ["llm", "diffusion-lora", "q4nx"]) {
    assert.match(page, new RegExp(`<TabsTrigger value="${tab}"`));
    // Each tab owns a panel, kept mounted so a running export survives a switch.
    assert.match(
      page,
      new RegExp(`<TabsContent value="${tab}" forceMount className=\\{TAB_PANEL\\}>`),
    );
  }
  assert.equal(page.match(/<ImageLoraExportCard /g)?.length, 1);
  assert.equal(page.match(/<Q4nxConvertCard /g)?.length, 1);
  // An image LoRA deep link opens on its tab.
  assert.match(page, /preselectImageLora \? "diffusion-lora" : "llm"/);
});
