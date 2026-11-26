// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { multiGpuPlacement } from "../src/lib/resolved-precision.ts";

test("no decision (one card) shows no multi-GPU placement", () => {
  assert.equal(multiGpuPlacement(null), null);
  assert.equal(multiGpuPlacement(undefined), null);
  assert.equal(multiGpuPlacement({ chosen: null }), null);
});

test("a CFG replica names its card and lists the rejected split", () => {
  const out = multiGpuPlacement({
    chosen: "cfg_replica",
    summary:
      "placement: one card plus a CFG replica on cuda:1: the default request is guided; cuda:0 runs its one-card 'group' plan",
    rejected: [
      { kind: "one_card", reason: "everything resident on cuda:0 needs 30 GiB" },
      { kind: "companion_split", reason: "the companion split would fit, but the CFG replica layout is predicted faster" },
    ],
  });
  assert.ok(out);
  assert.equal(out.label, "One card + CFG replica on cuda:1");
  assert.match(out.detail, /^Chosen: one card plus a CFG replica on cuda:1/);
  assert.match(out.detail, /Not chosen:\n- One card: everything resident/);
  assert.match(out.detail, /- Components split across cards: the companion split would fit/);
  assert.doesNotMatch(out.detail, /placement:/);
});

test("a companion split without rejected candidates shows only the summary", () => {
  const out = multiGpuPlacement({
    chosen: "companion_split",
    summary: "placement: companion split: text_encoder on cuda:1",
    rejected: [],
  });
  assert.deepEqual(out, {
    label: "Components split across cards",
    detail: "Chosen: companion split: text_encoder on cuda:1",
  });
});

test("an unknown kind still renders readably", () => {
  assert.equal(multiGpuPlacement({ chosen: "tensor_parallel" })?.label, "tensor parallel");
});
