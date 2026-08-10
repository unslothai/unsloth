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

test("other web routes keep the normal corner inset", () => {
  for (const pathname of ["/images", "/studio", "/settings"]) {
    assert.deepEqual(getToastOffsets(pathname, false), {
      default: { top: 12, right: 12 },
      mobile: { top: 16, right: 16 },
    });
  }
});

test("desktop routes also clear the titlebar", () => {
  assert.deepEqual(getToastOffsets("/chat", true), {
    default: { top: 86, right: 12 },
    mobile: { top: 86, right: 16 },
  });
  assert.deepEqual(getToastOffsets("/images", true), {
    default: { top: 46, right: 12 },
    mobile: { top: 50, right: 16 },
  });
});
