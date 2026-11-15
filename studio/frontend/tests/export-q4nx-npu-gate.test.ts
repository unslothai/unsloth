// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";

test("Q4NX export shows only with a supported NPU whose FastFlowLM runtime is ready", () => {
  const src = readSrc("features/export/export-page.tsx");
  assert.match(
    src,
    /const hasNpu = !!npuStatus\?\.supported && !!npuStatus\?\.ready;/,
  );
  // Every Q4NX surface and the export flag hang off that one gate.
  assert.match(src, /hasNpu && npuQ4nx && q4nxSourceSelected/);
  assert.equal((src.match(/\{hasNpu && <Q4nxConvertCard[ />]/g) ?? []).length, 2);
  assert.doesNotMatch(src, /q4nxSettings|useQ4nxExportStore/);
});
