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
