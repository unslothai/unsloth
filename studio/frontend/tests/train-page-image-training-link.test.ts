// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readdirSync } from "node:fs";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

test("the Train header links to the Images page with its Train panel selected", () => {
  const src = readSrc("features/studio/studio-page.tsx");
  const link = src.match(/<MediaPageLink\s+to="\/images"[\s\S]*?\/>/)?.[0];
  assert.ok(link, "no /images link in the Train header");
  assert.match(link, /label=\{t\("studio\.imageTraining"\)\}/);
  assert.match(link, /onNavigate=\{\(\) => setImagesPageMode\("train"\)\}/);
  assert.match(src, /useImageWorkflowStore\(\(s\) => s\.setPageMode\)/);
});

test("every locale names the image training link", async () => {
  const dir = new URL("../src/i18n/locales/", import.meta.url);
  const files = readdirSync(dir).filter((f) => f.endsWith(".ts"));
  assert.ok(files.length >= 14);
  for (const file of files) {
    const mod = await import(new URL(file, dir).href);
    const studio = (Object.values(mod)[0] as { studio: Record<string, unknown> }).studio;
    for (const key of ["imageTraining", "goToImageTraining"]) {
      assert.equal(typeof studio[key], "string", `${file} studio.${key}`);
      assert.ok((studio[key] as string).trim(), `${file} studio.${key} is empty`);
    }
  }
});
