// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { parseSearch, stringifySearch } from "../src/app/router-search.ts";

test("query text survives router parsing and redirects literally", () => {
  for (const q of ["null", "1.00", "true", '{"x":1}', '"hello"', "a+b & c"]) {
    const encoded = `?q=${encodeURIComponent(q)}&other=42`;
    const parsed = parseSearch(encoded);
    assert.equal(parsed.q, q);
    assert.equal(parsed.other, 42);
    assert.equal(parseSearch(stringifySearch(parsed)).q, q);
  }
});
