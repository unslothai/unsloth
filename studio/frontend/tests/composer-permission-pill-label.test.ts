// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";

test("the permission pill shows its icon only once collapsed", () => {
  const css = readSrc("index.css");
  assert.match(
    css,
    /\.composer-pill-btn\.composer-pill-permissions > \.composer-pill-glyph \{\s*display: none;/,
  );
  const compact = css.slice(css.indexOf('[data-pill-compact="true"] .composer-pill-btn'));
  assert.match(
    compact.slice(0, compact.indexOf("\n\t}\n")),
    /& \.composer-pill-glyph \{\s*display: inline-grid;/,
  );
});
