// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  DEFAULT_RESOLUTION,
  resolutionFor,
} from "../src/features/images/image-generation-defaults.ts";

test("the backend's recommended canvas is what seeds the size fields", () => {
  assert.deepEqual(resolutionFor({ recommendedCanvas: 512 }), {
    width: 512,
    height: 512,
  });
  assert.deepEqual(resolutionFor({ recommendedCanvas: 1024 }), {
    width: 1024,
    height: 1024,
  });
});

test("no opinion from the backend keeps 1024", () => {
  // null on unified memory and whenever the plan could not size the model; undefined on a backend
  // older than the field. All three have to keep the previous default rather than shrink on a
  // guess, so none of them may return 512.
  for (const value of [null, undefined, 0, Number.NaN, -512]) {
    assert.deepEqual(
      resolutionFor({ recommendedCanvas: value }),
      DEFAULT_RESOLUTION,
      String(value),
    );
  }
});
