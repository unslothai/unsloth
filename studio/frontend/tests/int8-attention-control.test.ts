// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  ATTENTION_SELECT_HINT,
  INT8_ATTENTION_HINT,
  INT8_ATTENTION_LABEL,
  INT8_ATTENTION_OPTIONS,
  int8AttentionSelection,
} from "../src/lib/int8-attention.ts";
import { type ResolvedControl, resolvedBadge, resolvedSeedKey } from "../src/lib/resolved-precision.ts";
import { readSrc } from "./helpers/kit.ts";

const PAGES = ["features/video/video-page.tsx", "features/images/images-page.tsx"] as const;

test("the control offers Auto, On and Off", () => {
  assert.equal(INT8_ATTENTION_LABEL, "INT8 attention");
  assert.deepEqual(INT8_ATTENTION_OPTIONS, [
    ["auto", "Auto"],
    ["on", "On"],
    ["off", "Off"],
  ]);
});

test("hints carry no em dash and the Attention hint points at the new control", () => {
  for (const text of [INT8_ATTENTION_HINT, ATTENTION_SELECT_HINT]) assert.ok(!text.includes("\u2014"));
  assert.match(ATTENTION_SELECT_HINT, /INT8 attention control below/);
  assert.doesNotMatch(ATTENTION_SELECT_HINT, /sage is INT8 attention \(/);
  assert.match(INT8_ATTENTION_HINT, /SageAttention \(INT8\) stays a separate choice/);
});

test("the select reseeds from the loaded build's resolved entry", () => {
  const auto: ResolvedControl = { value: "on", requested: "auto", source: "auto", status: "applied", reason: "measured faster" };
  assert.equal(int8AttentionSelection(auto), "auto");
  const on: ResolvedControl = { value: "on", requested: "on", source: "explicit", status: "applied", reason: "" };
  assert.equal(int8AttentionSelection(on), "on");
  const off: ResolvedControl = { value: "off", requested: "off", source: "explicit", status: "applied", reason: "" };
  assert.equal(int8AttentionSelection(off), "off");
  const declined: ResolvedControl = {
    value: "off",
    requested: "on",
    source: "explicit",
    status: "fell_back",
    reason: "the kernel failed the accuracy check",
  };
  assert.equal(int8AttentionSelection(declined), "off", "a declined On snaps to the engaged Off");
  // Older backends, sd.cpp and MLX loads do not report the key: leave the select alone.
  assert.equal(int8AttentionSelection(undefined), null);
  assert.equal(int8AttentionSelection(null), null);
});

test("the badge reads the resolved int8_attention entry", () => {
  const engaged = {
    value: "on",
    requested: "auto",
    source: "auto",
    status: "applied",
    reason: "measured accurate and faster here",
    engaged: true,
    impl: "cuda",
    modules: 48,
  } satisfies ResolvedControl;
  assert.deepEqual(resolvedBadge("int8_attention", engaged), {
    label: "Auto: ON",
    tone: "auto",
    tooltip: "measured accurate and faster here",
  });
  const declined: ResolvedControl = {
    value: "off",
    requested: "on",
    source: "explicit",
    status: "fell_back",
    reason: "the kernel failed the accuracy check",
    engaged: false,
    impl: null,
    modules: 0,
  };
  assert.equal(resolvedBadge("int8_attention", declined)?.label, "ON → OFF");
  assert.equal(resolvedBadge("int8_attention", declined)?.tone, "warn");
  assert.equal(resolvedBadge("int8_attention", undefined), null);
});

test("a new int8_attention outcome reseeds the select, an absent key does not break the key", () => {
  const base: Record<string, ResolvedControl> = {
    attention_backend: { value: "native", requested: null, source: "auto", status: "applied", reason: "" },
  };
  const withAuto = { ...base, int8_attention: { value: "on", requested: "auto", source: "auto", status: "applied", reason: "" } as ResolvedControl };
  const withOff = { ...base, int8_attention: { value: "off", requested: "off", source: "explicit", status: "applied", reason: "" } as ResolvedControl };
  assert.equal(typeof resolvedSeedKey(base), "string");
  assert.notEqual(resolvedSeedKey(withAuto), resolvedSeedKey(withOff));
});

for (const page of PAGES) {
  test(`${page} renders the control below Attention, sends it on load and reseeds it`, () => {
    const src = readSrc(page);
    const attention = src.indexOf('controlKey="attention_backend"');
    const int8 = src.indexOf('<ResolvedBadge status={status} controlKey="int8_attention" />');
    assert.ok(attention > 0 && int8 > attention, "badge-carrying INT8 attention select follows the Attention select");
    assert.match(src, /label=\{INT8_ATTENTION_LABEL\}[\s\S]{0,400}options=\{INT8_ATTENTION_OPTIONS\}/);
    assert.match(src, /useState<Int8AttentionChoice>\("auto"\)/);
    assert.match(src, /int8_attention: (controls\.)?int8Attention,/);
    assert.match(src, /int8_attention: advanced\.int8_attention,/);
    assert.match(src, /int8AttentionSelection\(record\.int8_attention\)/);
    assert.match(src, /hint=\{ATTENTION_SELECT_HINT\}/);
  });
}
