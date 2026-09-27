// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { getToastOffsets } from "../src/lib/toast-offset.ts";

test("web chat toasts clear the header and stay against the right edge", () => {
  assert.deepEqual(getToastOffsets("/chat", false), {
    default: { top: 52, right: 12 },
    mobile: { top: 52, right: 16 },
  });
  assert.deepEqual(getToastOffsets("/chat/thread", false), {
    default: { top: 52, right: 12 },
    mobile: { top: 52, right: 16 },
  });
});

test("web media toasts clear their workspace headers", () => {
  for (const pathname of ["/images", "/video"]) {
    assert.deepEqual(getToastOffsets(pathname, false), {
      default: { top: 52, right: 12 },
      mobile: { top: 52, right: 16 },
    });
  }
});

test("other web routes keep the normal corner inset", () => {
  for (const pathname of ["/studio", "/settings"]) {
    assert.deepEqual(getToastOffsets(pathname, false), {
      default: { top: 12, right: 12 },
      mobile: { top: 16, right: 16 },
    });
  }
});

test("desktop routes without page headers clear the titlebar", () => {
  assert.deepEqual(getToastOffsets("/settings", true), {
    default: { top: 46, right: 12 },
    mobile: { top: 50, right: 16 },
  });
});

test("desktop headers share the titlebar band", () => {
  // 10px under the header's controls: y 9-42 under macOS, 4-37 under the custom titlebar.
  for (const pathname of ["/chat", "/images", "/video", "/audio"]) {
    assert.deepEqual(getToastOffsets(pathname, true), {
      default: { top: 52, right: 12 },
      mobile: { top: 52, right: 16 },
    });
    assert.deepEqual(getToastOffsets(pathname, true, 1, true), {
      default: { top: 47, right: 12 },
      mobile: { top: 47, right: 16 },
    });
  }
});

test("a route that merely starts with a workspace name keeps the corner inset", () => {
  // The header routes are matched exactly, so a longer path that happens to share the
  // prefix must not inherit their clearance and drop 40px down a page with no header.
  for (const pathname of ["/chatty", "/images-old", "/videos", "/chatgpt"]) {
    assert.deepEqual(getToastOffsets(pathname, false), {
      default: { top: 12, right: 12 },
      mobile: { top: 16, right: 16 },
    });
  }
});

test("an unrecognised pathname falls back to the corner inset", () => {
  // The 404 shell paints no page header. This also covers a trailing-slash URL: the
  // router does not normalise it, so "/images/" rests as its own pathname and misses
  // the route, which is why it wants the no-header placement rather than the media one.
  for (const pathname of ["/unknown", "/images/", "/video/", ""]) {
    assert.deepEqual(getToastOffsets(pathname, false), {
      default: { top: 12, right: 12 },
      mobile: { top: 16, right: 16 },
    });
  }
});

test("offsets are pure, so a caller cannot poison the next lookup", () => {
  const first = getToastOffsets("/chat", false);
  first.default.top = -999;
  first.mobile.right = -999;
  assert.deepEqual(getToastOffsets("/chat", false), {
    default: { top: 52, right: 12 },
    mobile: { top: 52, right: 16 },
  });
});

test("the header offset follows the UI font size, the titlebar does not", () => {
  // The page header is 48px * the scale, so a fixed 52px top lands inside it
  // at the 20px setting.
  assert.deepEqual(getToastOffsets("/chat", false, 20 / 15), {
    default: { top: 69, right: 12 },
    mobile: { top: 69, right: 16 },
  });
  assert.deepEqual(getToastOffsets("/chat", true, 20 / 15), {
    default: { top: 69, right: 12 },
    mobile: { top: 69, right: 16 },
  });
  // A route with no header keeps its corner inset at any size.
  assert.deepEqual(getToastOffsets("/settings", false, 20 / 15), {
    default: { top: 12, right: 12 },
    mobile: { top: 16, right: 16 },
  });
});
