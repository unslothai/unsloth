// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

test("Model hub's Run uses the filled play glyph Library's play buttons use", () => {
  const card = readSrc("features/hub/catalog/download-card.tsx");
  const library = readSrc("features/library/components/library-cards.tsx");
  const filled =
    /<PlayIcon strokeWidth=\{2\} className="[^"]*\[&_path\]:fill-current"/;
  assert.match(library, filled);
  assert.match(
    card,
    /<PlayIcon strokeWidth=\{2\} className="\[&_path\]:fill-current" \/>\s*Run/,
  );
});
