// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// WebKitGTK stops drawing a Glass surface's own backdrop blur while a hidden (opacity 0) child of it
// still carries a backdrop-filter. The composer's and files panel's drop overlays sit inside those
// surfaces at opacity 0 until a file is dragged, so they may only blur while they show.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const THREAD = readSrc("components/assistant-ui/thread.tsx");
const SHARED = readSrc("features/chat/shared-composer.tsx");
const CSS = readSrc("index.css");

test("the composer's drop overlay blurs only in its shown states", () => {
  const base = /"(aui-composer-drop-overlay [^"]*)"/.exec(THREAD)?.[1] ?? "";
  assert.ok(base.includes("opacity-0"), "hidden at rest");
  const unconditional = base.split(/\s+/).filter((name) => /^backdrop-/.test(name));
  assert.deepEqual(unconditional, [], "no blur class without a state prefix");
  assert.ok(base.includes("group-data-[dragging=true]/dropzone:opacity-100"));
  assert.ok(base.includes("group-data-[dragging=true]/dropzone:backdrop-blur-sm"));
  assert.match(THREAD, /\(pageDragging \|\| nativeDragging\) && "opacity-100 backdrop-blur-sm"/);
});

test("the shared composer's drop overlay blurs only while dragging", () => {
  assert.match(SHARED, /\$\{dragging \? "opacity-100 backdrop-blur-sm" : "opacity-0"\}/);
  const overlay = /className=\{`(pointer-events-none absolute inset-0 z-20[^`]*)`\}/.exec(SHARED)?.[1] ?? "";
  assert.ok(overlay, "overlay found");
  assert.doesNotMatch(overlay.replace(/\$\{[^}]*\}/g, ""), /backdrop-/, "no blur outside the dragging branch");
});

test("the files panel's drop overlay blurs only while dragging, and a suppressed composer overlay drops its blur", () => {
  const rule = (start: string) => {
    const at = CSS.indexOf(start);
    assert.ok(at > 0, start);
    return CSS.slice(at, CSS.indexOf("}", at) + 1);
  };
  const rest = rule(".unsloth-files-panel-drop-overlay {");
  assert.match(rest, /opacity-0/);
  assert.doesNotMatch(rest, /backdrop-/);
  const shown = rule('.unsloth-files-panel[data-dragging="true"] .unsloth-files-panel-drop-overlay {');
  assert.match(shown, /@apply backdrop-blur-sm;/);
  assert.match(shown, /opacity: 1;/);
  const suppressed = rule('.aui-composer-root:has(.unsloth-files-panel[data-dragging="true"]) .aui-composer-drop-overlay {');
  assert.match(suppressed, /opacity: 0 !important;/);
  assert.match(suppressed, /\n\s*backdrop-filter: none !important;/);
});
