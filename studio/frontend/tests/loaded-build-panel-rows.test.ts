// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Native sd.cpp is the default image engine on GPU-less hosts. */

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const source = readSrc("features/images/images-page.tsx");

test("the Memory recipe row renders on an offload with no memory mode", () => {
  // The native engine reports memory_mode null while still offloading.
  const guard = source.match(
    /\{image\.memory_mode \|\|\s*\n\s*\(image\.offload_policy && image\.offload_policy !== "none"\) \?/,
  );
  assert.ok(guard, "the Memory row must render when EITHER field conveys placement");
});

test("sd.cpp attention is not reported as Native SDPA", () => {
  const start = source.search(/<BuildRow\s+label="Attention"/);
  assert.ok(start >= 0, "the Loaded-build panel must keep its Attention row");
  const attention = source.slice(start, start + 700);
  assert.ok(
    attention.includes("isNativeEngineStatus(status)"),
    "the fallback must distinguish the native engine before naming SDPA",
  );
  assert.ok(attention.includes("sd.cpp"), "the native arm needs its own label");
  assert.ok(attention.includes('"Native SDPA"'), "diffusers keeps its label");
});
