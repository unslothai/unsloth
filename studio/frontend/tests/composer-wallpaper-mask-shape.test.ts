// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";

const css = readSrc("index.css");
const start = css.indexOf("/* Masks the thread only inside the composer's rounded box");
const rule = css.slice(start, css.indexOf("\n}\n", start));

test("the wallpaper mask cuts the thread at the composer's top edge", () => {
  assert.ok(start >= 0);
  assert.match(
    rule,
    /#000 calc\(100% - var\(--clip\)\), transparent calc\(100% - var\(--clip\) \+ var\(--fade\)\)/,
  );
});

test("text beside the composer and outside its top corners stays visible", () => {
  assert.equal(rule.match(/radial-gradient/g)?.length, 2);
  assert.match(rule, /left 0 bottom var\(--below\)/);
  assert.match(rule, /right 0 bottom var\(--below\)/);
  assert.match(rule, /left var\(--thread-composer-left, 0px\) bottom calc\(var\(--clip\) - var\(--radius\)\)/);
  assert.match(rule, /right var\(--thread-composer-right, 0px\) bottom calc\(var\(--clip\) - var\(--radius\)\)/);
});

test("the composer dock publishes every box variable the mask reads", () => {
  const thread = readSrc("components/assistant-ui/thread.tsx");
  for (const name of ["clip", "bottom", "left", "right", "radius", "fade"]) {
    assert.match(thread, new RegExp(`"--thread-composer-${name}"`));
  }
  assert.match(thread, /"calc\(2px \* var\(--ui-space-scale, 1\)\)"/);
});

test("the mask measures the composer after its insets are applied", () => {
  const thread = readSrc("components/assistant-ui/thread.tsx");
  assert.match(
    thread,
    /"--composer-dock-inset-right", `\$\{inset\.right\}px`\);\n(?:\s*\/\/.*\n)?\s*syncComposerClip\(\);\n\s*\};/,
  );
});
