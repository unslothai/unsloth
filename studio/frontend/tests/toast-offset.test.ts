// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { getToastOffsets } from "../src/lib/toast-offset.ts";

test("web chat toasts clear the header and stay against the right edge", () => {
  assert.deepEqual(getToastOffsets("/chat", false, false), {
    default: { top: 52, right: 12 },
    mobile: { top: 52, right: 16 },
  });
  assert.deepEqual(getToastOffsets("/chat/thread", false, false), {
    default: { top: 52, right: 12 },
    mobile: { top: 52, right: 16 },
  });
});

test("web media toasts clear their workspace headers", () => {
  for (const pathname of ["/images", "/video"]) {
    assert.deepEqual(getToastOffsets(pathname, false, false), {
      default: { top: 52, right: 12 },
      mobile: { top: 52, right: 16 },
    });
  }
});

test("other web routes keep the normal corner inset", () => {
  for (const pathname of ["/studio", "/settings"]) {
    assert.deepEqual(getToastOffsets(pathname, false, false), {
      default: { top: 12, right: 12 },
      mobile: { top: 16, right: 16 },
    });
  }
});

test("desktop routes without page headers clear the titlebar", () => {
  assert.deepEqual(getToastOffsets("/settings", true, false), {
    default: { top: 46, right: 12 },
    mobile: { top: 50, right: 16 },
  });
});

test("custom-titlebar desktop headers clear both titlebar bands", () => {
  for (const pathname of ["/chat", "/images", "/video"]) {
    assert.deepEqual(getToastOffsets(pathname, true, true), {
      default: { top: 86, right: 12 },
      mobile: { top: 86, right: 16 },
    });
  }
});

test("macOS desktop headers overlay the native titlebar", () => {
  for (const pathname of ["/chat", "/images", "/video"]) {
    assert.deepEqual(getToastOffsets(pathname, true, false), {
      default: { top: 52, right: 12 },
      mobile: { top: 52, right: 16 },
    });
  }
});

test("a route that merely starts with a workspace name keeps the corner inset", () => {
  // The header routes are matched exactly, so a longer path that happens to share the
  // prefix must not inherit their clearance and drop 40px down a page with no header.
  for (const pathname of ["/chatty", "/images-old", "/videos", "/chatgpt"]) {
    assert.deepEqual(getToastOffsets(pathname, false, false), {
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
    assert.deepEqual(getToastOffsets(pathname, false, false), {
      default: { top: 12, right: 12 },
      mobile: { top: 16, right: 16 },
    });
  }
});

test("a custom titlebar is ignored off the desktop app", () => {
  // shouldUseCustomWindowTitlebar() cannot return true while isTauri is false, but the
  // signature allows the pair, and there is no titlebar to clear in a browser.
  for (const pathname of ["/chat", "/studio"]) {
    assert.deepEqual(
      getToastOffsets(pathname, false, true),
      getToastOffsets(pathname, false, false),
    );
  }
});

test("offsets are pure, so a caller cannot poison the next lookup", () => {
  const first = getToastOffsets("/chat", false, false);
  first.default.top = -999;
  first.mobile.right = -999;
  assert.deepEqual(getToastOffsets("/chat", false, false), {
    default: { top: 52, right: 12 },
    mobile: { top: 52, right: 16 },
  });
});
