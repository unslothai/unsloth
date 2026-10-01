// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

// The UI font size scales text and spacing, never a chatbox's width. Its caps and the gutters
// around it use --ui-layout-scale, which is the interface scale alone.

test("the layout scale is the interface scale, not the font size", () => {
  const css = readSrc("index.css");
  assert.match(css, /--ui-layout-scale: var\(--ui-interface-scale, 1\);/);
});

test("chatbox caps and gutters never read the font-scaled spacing", () => {
  const thread = readSrc("components/assistant-ui/thread.tsx");
  const chatPage = readSrc("features/chat/chat-page.tsx");
  const appearance = readSrc("features/settings/stores/appearance-custom-store.ts");
  for (const [file, source, needles] of [
    ["thread.tsx", thread, [
      '"var(--custom-chat-max-width, calc(48rem * var(--ui-layout-scale, 1)))"',
      '"calc(var(--thread-max-width) - 1.5rem * var(--ui-layout-scale, 1))"',
      "var(--thread-content-max-width,calc(46rem*var(--ui-layout-scale,1)))",
      "scroll-smooth px-[calc(20px*var(--ui-layout-scale,1))]",
      "unsloth-composer-dock-inner relative px-[calc(20px*var(--ui-layout-scale,1))]",
    ]],
    ["chat-page.tsx", chatPage, [
      "max-w-[calc(44rem*var(--ui-layout-scale,1))]",
      "max-w-[var(--custom-chat-max-width,calc(48rem*var(--ui-layout-scale,1)))]",
      "px-[calc(20px*var(--ui-layout-scale,1))] md:px-[calc(30px*var(--ui-layout-scale,1))]",
    ]],
    ["appearance-custom-store.ts", appearance, [
      '"calc(72rem * var(--ui-layout-scale, 1))"',
    ]],
  ] as const) {
    for (const needle of needles) assert.ok(source.includes(needle), `${file}: ${needle}`);
  }
  assert.equal(/max-w-\[calc\(44rem\*var\(--ui-space-scale/.test(chatPage), false);
});

test("media prompt rails and their gutters ignore the font size", () => {
  for (const file of ["features/images/images-page.tsx", "features/audio/audio-page.tsx", "features/video/video-page.tsx"]) {
    const source = readSrc(file);
    assert.ok(
      source.includes("px-[calc(40px*var(--ui-layout-scale,1))] max-sm:px-[calc(20px*var(--ui-layout-scale,1))]"),
      `${file} prompt panel gutter`,
    );
    assert.equal(source.includes("calc(408px*var(--ui-space-scale,1))"), false, `${file} rail width`);
  }
  // The resize handle sits on the rail's edge, so it falls back to the same width.
  assert.equal(readSrc("components/media-rail-resize-handle.tsx").includes("--ui-space-scale"), false);
  const hook = readSrc("hooks/use-media-rail-width.ts");
  assert.match(hook, /rail\.width \* rail\.scale/);
  assert.equal(/useUiSpaceScale/.test(hook), false);
});
