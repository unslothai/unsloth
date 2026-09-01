// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { normalizeSizeInputDraft } from "../src/features/settings/components/size-input-value.ts";

test("an out-of-range draft is reconciled to the active boundary", () => {
  assert.deepEqual(normalizeSizeInputDraft("250", { min: 25, max: 200 }), {
    draft: "200",
    value: 200,
  });
});
