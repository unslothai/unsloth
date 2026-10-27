// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";

test("only the Linux desktop app drops the full-size media shadow", () => {
  const css = readSrc("index.css");
  const rules = [
    ...css.matchAll(/([^{}]*\.media-stage-shadow[^{}]*)\{([^}]*)\}/g),
  ];
  assert.equal(rules.length, 1);
  assert.equal(
    rules[0][1].trim().split("\n").pop()?.trim(),
    "html.tauri.render-linux .media-stage-shadow",
  );
  assert.equal(rules[0][2].trim(), "box-shadow: none;");
});

test("every full-size media element with a shadow on Images and Video opts in", () => {
  for (const file of [
    "features/images/images-page.tsx",
    "features/video/video-page.tsx",
  ]) {
    const src = readSrc(file);
    const shadowed = [
      ...src.matchAll(/className="([^"]*object-contain[^"]*shadow-sm[^"]*)"/g),
    ];
    assert.equal(shadowed.length, 2, file);
    for (const [, cls] of shadowed)
      assert.match(cls, /\bmedia-stage-shadow\b/, `${file}: ${cls}`);
  }
});

test("the Images and Video prompt boxes hand keystrokes to the page as a transition", () => {
  for (const file of [
    "features/images/images-page.tsx",
    "features/video/video-page.tsx",
  ]) {
    const src = readSrc(file);
    const at = src.indexOf('data-type-to-activate="prompt"');
    assert.match(src.slice(at - 40, at), /<DeferredTextarea\s*$/, file);
    assert.match(src, /onValueChange=\{setPrompt\}/, file);
  }
  const component = readSrc("components/deferred-textarea.tsx");
  assert.match(component, /startTransition\(\(\) => onValueChange\(next\)\)/);
});
