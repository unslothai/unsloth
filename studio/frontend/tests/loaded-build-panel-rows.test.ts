// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Two rows of the Loaded-build panel, checked against the source that renders them.
 *
 * Both cases are the native sd.cpp engine, which is the DEFAULT image path on a host with no
 * usable GPU -- so a row that is only correct for diffusers is wrong for most first runs.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import test from "node:test";

const source = readFileSync(
  fileURLToPath(new URL("../src/features/images/images-page.tsx", import.meta.url)),
  "utf8",
);

test("the Memory recipe row renders on an offload with no memory mode", () => {
  // The native engine reports memory_mode null (it has no torchao path to choose one for) while
  // still recording an active offload, so gating the row on memory_mode alone hid the offload
  // policy on exactly the configuration the row was extended to expose.
  const guard = source.match(
    /\{image\.memory_mode \|\|\s*\n\s*\(image\.offload_policy && image\.offload_policy !== "none"\) \?/,
  );
  assert.ok(guard, "the Memory row must render when EITHER field conveys placement");
});

test("sd.cpp attention is not reported as Native SDPA", () => {
  // Its attention is chosen by native flags, not by the diffusers/PyTorch dispatcher.
  const attention = source.slice(
    source.indexOf('label="Attention"'),
    source.indexOf('label="Attention"') + 700,
  );
  assert.ok(
    attention.includes("isNativeEngineStatus(status)"),
    "the fallback must distinguish the native engine before naming SDPA",
  );
  assert.ok(attention.includes("sd.cpp"), "the native arm needs its own label");
  assert.ok(attention.includes('"Native SDPA"'), "diffusers keeps its label");
});
