// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const dialog = readSrc("features/settings/settings-dialog.tsx");

test("settings pages pad both sides alike", () => {
  assert.match(dialog, /overflow-y-auto px-12 py-6/);
});

test("the pane's scrollbar is drawn on the dialog's very edge", () => {
  assert.match(
    dialog,
    /<ListScrollbar\s+target=\{mainScrollRef\}\s+flush=\{true\}\s+inset=\{2\}\s+thumbClassName="settings-scroll-thumb"/,
  );
  const css = readSrc("index.css");
  assert.match(
    css,
    /\[data-settings-main-scroll\] \{\s*scrollbar-width: none !important;/,
  );
  assert.match(
    css,
    /\[data-settings-main-scroll\]::-webkit-scrollbar \{\s*display: none;/,
  );
  // Same look as the hover-scrollbar it replaced; only its position moved.
  assert.match(
    css,
    /\.settings-scroll-thumb \{\s*width: 6px;\s*background: oklch\(0\.5 0 0 \/ 0\.4\);/,
  );
  assert.match(
    css,
    /\.dark \.settings-scroll-thumb \{\s*background: oklch\(0\.72 0 0 \/ 0\.4\);/,
  );
});
