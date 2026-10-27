// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { WORKFLOW_TABS } from "../src/features/images/workflows.ts";
import { AiPaintbrushIcon } from "../src/lib/ai-paintbrush-icon.ts";
import { readSrc } from "./helpers/kit.ts";

const PRESETS = readSrc(
  "features/generation-presets/media-generation-preset-control.tsx",
);

test("preset menu actions use Hugeicons' upload and clipboard-paste", () => {
  assert.match(
    PRESETS,
    /<HugeiconsIcon icon=\{Upload01Icon\}[^>]*\/>\s*Load settings from an image/,
  );
  assert.match(
    PRESETS,
    /<HugeiconsIcon icon=\{ClipboardPasteIcon\}[^>]*\/>\s*Paste generation parameters/,
  );
  assert.doesNotMatch(PRESETS, /ImageUp|ClipboardPaste,/);
});

test("Inpaint uses the vendored AI paintbrush", () => {
  assert.equal(
    WORKFLOW_TABS.find((tab) => tab.id === "inpaint")?.icon,
    AiPaintbrushIcon,
  );
  assert.equal(AiPaintbrushIcon.length, 3);
});

test("the save row shares the preset list's 8px sides", () => {
  assert.match(PRESETS, /border-t border-border\/60 bg-muted\/20 px-2 py-3/);
  assert.match(PRESETS, /className="mb-2 block px-1 text-xs font-medium"/);
  assert.match(
    PRESETS,
    /max-h-48 min-h-0 overflow-y-auto overscroll-contain p-2/,
  );
});

test("component file fields are the selects' 36px", () => {
  const page = readSrc("features/images/images-page.tsx");
  assert.match(
    page,
    /rows=\{1\}[\s\S]{0,200}className="min-h-9 overflow-hidden rounded-\[18px\] py-\[calc\(\(var\(--spacing\)\*9-2px-1lh\)\/2\)\] font-mono text-xs leading-5"/,
  );
  assert.match(
    page,
    /onChange=\{\(e\) => onValueChange\(e\.target\.value\)\}\s*className="h-9 font-mono text-xs"/,
  );
});
