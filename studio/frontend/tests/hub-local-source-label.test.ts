// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { localSourceLabel } = await import(
  "../src/features/hub/inventory/view-models.ts"
);
const { LOCAL_MODEL_SOURCES } = await import(
  "../src/features/hub/inventory/constants.ts"
);

test("every known local source has its own label, not the generic fallback", () => {
  for (const source of LOCAL_MODEL_SOURCES) {
    assert.notEqual(
      localSourceLabel(source),
      "Local model",
      `${source} falls through to the generic label`,
    );
  }
});
