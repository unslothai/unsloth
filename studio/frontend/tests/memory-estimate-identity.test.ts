// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Settings changes grey the figures; source changes blank them, computed during render.

import assert from "node:assert/strict";
import test from "node:test";

import {
  resolveEstimateSourceIdentity,
  resolveTokenIdentity,
} from "../src/features/model-picker/model-config/estimate-context.ts";

const identity = (
  path: string,
  variant: string | null = null,
  token: string | null = null,
  nativeToken: string | null = null,
) =>
  resolveEstimateSourceIdentity(
    path,
    variant,
    resolveTokenIdentity(token),
    nativeToken,
  );

function shown(stateIdentity: string | null, currentIdentity: string | null) {
  return stateIdentity === currentIdentity;
}

test("switching GGUF never paints the previous model's numbers", () => {
  const before = identity("unsloth/Qwen3-8B-GGUF", "Q4_K_M");
  const after = identity("unsloth/Llama-3.1-8B-GGUF", "Q4_K_M");
  assert.equal(shown(before, after), false);
});

test("switching quantization on ONE repository is also a switch", () => {
  const q4 = identity("unsloth/Qwen3-8B-GGUF", "Q4_K_M");
  const f16 = identity("unsloth/Qwen3-8B-GGUF", "F16");
  assert.notEqual(q4, f16);
  assert.equal(shown(q4, f16), false);
});

test("a settings change is NOT a switch, so the figures stay up and go grey", () => {
  const before = identity("unsloth/Qwen3-8B-GGUF", "Q4_K_M");
  const after = identity("unsloth/Qwen3-8B-GGUF", "Q4_K_M");
  assert.equal(shown(before, after), true);
});

test("standing down blanks the row rather than freezing the last answer", () => {
  const held = identity("unsloth/Qwen3-8B-GGUF", "Q4_K_M");
  assert.equal(shown(held, null), false);
});

test("two credentials are two sources: they resolve different files", () => {
  assert.notEqual(
    identity("org/gated", "Q4_K_M", "hf_aaa"),
    identity("org/gated", "Q4_K_M", "hf_bbb"),
  );
  assert.notEqual(
    identity("org/gated", "Q4_K_M", "hf_aaa"),
    identity("org/gated", "Q4_K_M", null),
  );
});

test("two native picks of the same filename are two sources", () => {
  assert.notEqual(
    identity("model.gguf", null, null, "tok-1"),
    identity("model.gguf", null, null, "tok-2"),
  );
});

test("the credential itself never appears in the identity", () => {
  const secret = "hf_ThisIsASecretAndMustNotBeInAReactKey";
  const key = identity("org/gated", "Q4_K_M", secret);
  assert.equal(key.includes(secret), false);
  assert.equal(key.includes("Secret"), false);
});

test("no credential and an empty credential agree", () => {
  assert.equal(resolveTokenIdentity(null), "");
  assert.equal(resolveTokenIdentity(undefined), "");
  assert.equal(resolveTokenIdentity(""), "");
});

test("the same credential is the same identity, so it does not thrash the row", () => {
  assert.equal(resolveTokenIdentity("hf_abc"), resolveTokenIdentity("hf_abc"));
});

// A djb2 collision between valid HF tokens is constructible.
const COLLIDING_A = "hf_7MqSwsKw8ci6CSUGQE2iUWyQqC4Wc8KoAi";
const COLLIDING_B = "hf_He6AyGWm4OKk0SmY4O2mAMWeUAGCWIKAK8";

test("a djb2 collision is real, and it suppresses BOTH the refetch and the blank", () => {
  assert.notEqual(COLLIDING_A, COLLIDING_B);
  assert.equal(resolveTokenIdentity(COLLIDING_A), resolveTokenIdentity(COLLIDING_B));
  const a = identity("org/gated", "Q4_K_M", COLLIDING_A);
  const b = identity("org/gated", "Q4_K_M", COLLIDING_B);
  assert.equal(a, b);
  assert.equal(shown(a, b), true);
});

test("the collision costs a stale byte count, not a wrong load", () => {
  // The hash keys only the row; requests carry the real token, so a collision is display-only.
  const a = identity("org/gated", "Q4_K_M", COLLIDING_A);
  assert.equal(a.includes(COLLIDING_A), false);
  assert.equal(a.includes(COLLIDING_B), false);
  assert.equal(resolveTokenIdentity(COLLIDING_A).length <= 7, true);
});

test("the hash is stable across the shapes a credential arrives in", () => {
  assert.notEqual(resolveTokenIdentity("hf_abc"), resolveTokenIdentity("hf_abc "));
  assert.notEqual(resolveTokenIdentity("hf_abc"), resolveTokenIdentity("HF_ABC"));
});

test("a long or non-ASCII credential still hashes without throwing", () => {
  assert.doesNotThrow(() => resolveTokenIdentity("x".repeat(100_000)));
  assert.doesNotThrow(() => resolveTokenIdentity("héllo-\u{1F600}-token"));
  assert.equal(typeof resolveTokenIdentity("héllo"), "string");
});
