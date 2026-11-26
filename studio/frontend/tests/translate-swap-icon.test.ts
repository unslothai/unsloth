// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const page = readFileSync(
  new URL("../src/features/translate/translate-page.tsx", import.meta.url),
  "utf8",
);

test("Translate's swap button draws the Keyline swap icon on both layouts", () => {
  assert.doesNotMatch(page, /ArrowLeftRightIcon/);
  assert.equal(page.match(/<SwapArrowsIcon /g)?.length, 2);
});

test("the model chip and the options button share one size and one hover", () => {
  assert.match(
    page,
    /unsloth-model-selector-trigger h-\[calc\(40px\*var\(--ui-space-scale,1\)\)\]/,
  );
  assert.match(
    page,
    /cn\(ICON_ACTION, "size-\[calc\(36px\*var\(--ui-space-scale,1\)\)\]"\)/,
  );
  const css = readFileSync(
    new URL("../src/index.css", import.meta.url),
    "utf8",
  );
  assert.match(
    css,
    /\.dark \.translate-page \.unsloth-thinking-pill:is\(:hover:not\(:disabled\), \[data-state="open"\]\) \{\s*background-color: color-mix\(in oklab, var\(--muted\) 50%, transparent\);/,
  );
});

test("option descriptions stay in the label column, beside the toggle", () => {
  const row = page.match(/function OptionRow[\s\S]*?\n}\n/)?.[0] ?? "";
  assert.match(row, /<div className="grid min-w-0 gap-2">\s*<FieldLabel/);
  assert.match(
    row,
    /\{description && <p className=\{HINT\}>\{description\}<\/p>\}\s*<\/div>\s*<div className="flex h-\[1lh\] shrink-0/,
  );
  const en = readFileSync(
    new URL("../src/i18n/locales/en.ts", import.meta.url),
    "utf8",
  );
  assert.match(
    en,
    /autoDescription: "Starts automatically when model is loaded\.",/,
  );
  assert.match(
    en,
    /showPromptDescription: "See exact instructions sent to the model\.",/,
  );
});
