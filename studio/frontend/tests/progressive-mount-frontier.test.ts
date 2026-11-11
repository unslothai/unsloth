// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A reader who keeps wheeling or paging up right after opening a long chat reaches the top of the
// mounted rows. Every wheel or key event re-arms the 300 ms input pause, so without an exception
// the window never widens while they keep going: they sit against the frontier until they stop.

import assert from "node:assert/strict";
import test from "node:test";

import { watchProgressiveScroll } from "../src/components/assistant-ui/progressive-mount-controller.ts";
import { readText } from "./helpers/kit.ts";

const GLUE = readText("../src/components/assistant-ui/progressive-messages.tsx").replace(/\/\*[\s\S]*?\*\//g, "").replace(/^[ \t]*\/\/.*$/gm, "");

function harness() {
  let clock = 1000;
  const viewport = new EventTarget();
  const activity = watchProgressiveScroll(viewport, () => clock);
  return {
    activity,
    wheel: () => viewport.dispatchEvent(new Event("wheel")),
    advance: (ms: number) => { clock += ms; },
  };
}

test("a reader stopped at the mounted top does not hold the widening off while wheeling", () => {
  const h = harness();
  h.advance(400);
  for (let i = 0; i < 20; i += 1) {
    h.wheel();
    h.advance(16);
    assert.equal(h.activity.blocked(false), true, "mid-list wheel input still pauses the widening");
    assert.equal(h.activity.blocked(false, true), false, "at the mounted top, the widening proceeds");
  }
});

test("the exception never applies to a held pointer or before any input", () => {
  const h = harness();
  assert.equal(h.activity.blocked(false, true), true, "the initial tail still paints first");
  h.advance(400);
  h.wheel();
  assert.equal(h.activity.blocked(true, true), true, "a scrollbar drag maps the pointer to scrollTop");
});

test("the widening loop tells the watcher whether the reader is at the mounted top", () => {
  assert.match(GLUE, /blocked\(inputPointerHeld\(\), atMountedTop\(viewportRef\.current\)\)/);
  assert.match(GLUE, /scrollTop <= 1 && viewport\.scrollHeight > viewport\.clientHeight/);
});
