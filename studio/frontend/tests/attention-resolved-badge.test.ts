// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  type ResolvedControl,
  engagedAttentionBackend,
  resolvedBadge,
  resolvedRequestSelectValue,
} from "../src/lib/resolved-precision.ts";
import { readSrc } from "./helpers/kit.ts";

const PAGES = ["features/video/video-page.tsx", "features/images/images-page.tsx"] as const;
const OPTIONS = ["auto", "native", "cudnn", "flash3", "sage"] as const;
const toOption = (v: string) => OPTIONS.find((o) => o === v || `_native_${o}` === v) ?? null;

const sageMissing: ResolvedControl = {
  value: "native",
  requested: "sage",
  source: "explicit",
  status: "fell_back",
  reason: "sageattention is not installed and the Hugging Face kernels-hub build could not be loaded (no build)",
  engaged: "native",
  modules: 57,
  routed: { native: 57 },
};

test("a sage pick that fell back shows what ran and why in the existing badge", () => {
  const info = resolvedBadge("attention_backend", sageMissing);
  assert.deepEqual(info, {
    label: "SAGE → NATIVE",
    tone: "warn",
    tooltip:
      "You requested SAGE; NATIVE was used because sageattention is not installed and the Hugging Face kernels-hub build could not be loaded (no build).",
  });
});

test("a pick the platform cannot run reads the same way", () => {
  const info = resolvedBadge("attention_backend", {
    ...sageMissing,
    status: "unsupported",
    reason: "SageAttention needs an NVIDIA CUDA GPU and this load runs on Apple Silicon (MPS)",
  });
  assert.equal(info?.label, "SAGE → NATIVE");
  assert.equal(info?.tone, "warn");
  assert.match(info?.tooltip ?? "", /Apple Silicon/);
});

test("INT8 On over a sage pick names the INT8 kernel", () => {
  const info = resolvedBadge("attention_backend", {
    value: "unsloth_int8",
    requested: "sage",
    source: "explicit",
    status: "fell_back",
    reason: "INT8 attention is On, so Studio's INT8 kernel runs 57 attention modules",
    engaged: "unsloth_int8",
    modules: 57,
  });
  assert.equal(info?.label, "SAGE → INT8");
  assert.match(info?.tooltip ?? "", /57 attention modules/);
});

test("an honored pick shows no badge; the hub build is SageAttention", () => {
  const honored: ResolvedControl = {
    value: "sage_hub",
    requested: "sage",
    source: "explicit",
    status: "applied",
    reason: "requested",
    engaged: "sage_hub",
    modules: 57,
    routed: { sage_hub: 57 },
  };
  assert.equal(resolvedBadge("attention_backend", honored), null);
  const auto = resolvedBadge("attention_backend", { ...honored, source: "auto", requested: null, value: "unsloth_int8", reason: "unsloth_int8 attention upgrade" });
  assert.equal(auto?.label, "Auto: INT8");
});

test("the Attention select keeps the user's pick when it fell back", () => {
  assert.equal(resolvedRequestSelectValue(sageMissing, toOption), "sage");
  assert.equal(resolvedRequestSelectValue({ ...sageMissing, status: "applied", value: "sage" }, toOption), "sage");
  assert.equal(resolvedRequestSelectValue({ ...sageMissing, source: "auto", requested: null }, toOption), "auto");
  assert.equal(resolvedRequestSelectValue(undefined, toOption), null);
});

test("the build summary names the backend the modules run, not the load's pick", () => {
  assert.equal(engagedAttentionBackend({ attention_backend: "unsloth_int8", resolved: { attention_backend: { ...sageMissing, value: "_native_flash" } } }), "_native_flash");
  assert.equal(engagedAttentionBackend({ attention_backend: "sage", resolved: { attention_backend: sageMissing } }), null);
  assert.equal(engagedAttentionBackend({ attention_backend: "sage_hub", resolved: null }), "sage_hub");
  assert.equal(engagedAttentionBackend({ attention_backend: null }), null);
});

for (const page of PAGES) {
  test(`${page} reads the attention badge, select and summary off the resolved entry`, () => {
    const src = readSrc(page);
    assert.match(src, /<ResolvedBadge status=\{status\} controlKey="attention_backend" \/>/);
    assert.match(src, /resolvedRequestSelectValue\(record\.attention_backend,/);
    assert.match(src, /engagedAttentionBackend\(status\)/);
  });
}
