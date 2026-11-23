// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";

test("the Windows thread does not rubber-band at its end", () => {
  assert.match(
    readSrc("index.css"),
    /:root\.client-windows \.aui-thread-viewport \{\s*overscroll-behavior-y: none;\s*\}/,
  );
});
