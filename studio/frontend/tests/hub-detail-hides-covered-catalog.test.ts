// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrcAsync } from "./helpers/kit.ts";

// #17004: a background image makes .hub-canvas transparent, so the model page showed the
// catalog through it.
test("an open model page hides the catalog it covers", async () => {
  const css = await readSrcAsync("index.css");
  assert.match(css, /\[data-wallpaper-ready\][^{]*\.hub-page \.hub-canvas[^{]*\{ background: transparent; \}/);

  const page = await readSrcAsync("features/hub/hub-page.tsx");
  assert.match(page, /className="hub-canvas absolute inset-0 z-20/);
  assert.match(page, /hub-canvas z-20 [^"]*max-lg:absolute max-lg:inset-0/);
  assert.match(
    page,
    /detailOpen && \(splitMode \? "max-lg:invisible" : "invisible"\)/,
  );
  // An explicit `visible` on a descendant would repaint it through the hidden pane.
  const catalog = await readSrcAsync("features/hub/catalog/models-catalog.tsx");
  assert.doesNotMatch(catalog, /\bvisible\b(?![-:])"/);
  assert.match(catalog, /!discoverActive && "pointer-events-none invisible"/);
  assert.match(catalog, /!downloadedActive && "pointer-events-none invisible"/);
});
