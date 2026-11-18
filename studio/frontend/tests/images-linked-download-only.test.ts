// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const page = readSrc("features/images/images-page.tsx");

test("a download-only pick of a linked model is refused before it moves the page or reaches this machine's downloads", () => {
  const body = page.slice(page.indexOf("const handleModelSelect = useCallback("));
  const refuse = body.indexOf("if (downloadOnlyPick && machine)");
  assert.ok(refuse >= 0 && refuse < body.indexOf("switchImagesMachine(machine);"));
});
