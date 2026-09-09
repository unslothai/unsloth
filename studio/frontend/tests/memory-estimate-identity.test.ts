// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What useMemoryEstimate is allowed to leave on screen.
//
// Two different rules. A settings change GREYS the figures and keeps them up, so a
// slider drag does not strobe the row; a SOURCE change blanks them, because one
// model's footprint under another's name is worse than none. Both come from
// `resolveEstimateSourceIdentity`, the narrower key, computed DURING RENDER rather than
// read from a ref the effect updates after paint -- effects run after React paints, so
// a direct switch between two GGUFs showed the previous model's numbers for a frame.
//
// The credential is part of the source, hashed rather than carried. That hash is 32
// bits, and the last test here is what that costs.

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

/** The guard the hook runs during render: state belonging to another source is not
 *  returned at all, not even for the frame before the effect clears it. */
function shown(stateIdentity: string | null, currentIdentity: string | null) {
  return stateIdentity === currentIdentity;
}

test("switching GGUF never paints the previous model's numbers", () => {
  const before = identity("unsloth/Qwen3-8B-GGUF", "Q4_K_M");
  const after = identity("unsloth/Llama-3.1-8B-GGUF", "Q4_K_M");
  // The render that first names the new model still holds the old model's state.
  assert.equal(shown(before, after), false);
});

test("switching quantization on ONE repository is also a switch", () => {
  // modelPath is identical across this change while the weights roughly quadruple.
  const q4 = identity("unsloth/Qwen3-8B-GGUF", "Q4_K_M");
  const f16 = identity("unsloth/Qwen3-8B-GGUF", "F16");
  assert.notEqual(q4, f16);
  assert.equal(shown(q4, f16), false);
});

test("a settings change is NOT a switch, so the figures stay up and go grey", () => {
  // Context, KV dtype, slots, pins: none of them select a different file, so the
  // source identity is unchanged and the hook keeps the numbers with `stale` set.
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
  // And clearing the credential is a switch too.
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

// ---------------------------------------------------------------------------
// The token hash

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

// The 32-bit question, answered rather than assumed. These two are both well-formed
// HF tokens ("hf_" plus 34 base62 characters) found by a birthday search over djb2;
// the point is that a collision is CONSTRUCTIBLE, not that one is likely.
const COLLIDING_A = "hf_7MqSwsKw8ci6CSUGQE2iUWyQqC4Wc8KoAi";
const COLLIDING_B = "hf_He6AyGWm4OKk0SmY4O2mAMWeUAGCWIKAK8";

test("a djb2 collision is real, and it suppresses BOTH the refetch and the blank", () => {
  assert.notEqual(COLLIDING_A, COLLIDING_B);
  assert.equal(resolveTokenIdentity(COLLIDING_A), resolveTokenIdentity(COLLIDING_B));
  // Same hash, so the source identity matches: the render-time guard cannot tell the
  // two apart, and the effect key does not change either, so nothing re-fetches.
  const a = identity("org/gated", "Q4_K_M", COLLIDING_A);
  const b = identity("org/gated", "Q4_K_M", COLLIDING_B);
  assert.equal(a, b);
  assert.equal(shown(a, b), true);
});

test("the collision costs a stale byte count, not a wrong load", () => {
  // Worth stating in a test because it is the reason this is documented rather than
  // fixed. The hash keys the ROW only. The load itself, and the estimate REQUEST when
  // one is made, both carry the real credential, so a collision can leave last
  // token's figures on screen and can never send the wrong token anywhere.
  const a = identity("org/gated", "Q4_K_M", COLLIDING_A);
  assert.equal(a.includes(COLLIDING_A), false);
  assert.equal(a.includes(COLLIDING_B), false);
  // The bound is two tokens compared per tab, so ~2^-32 per swap.
  assert.equal(resolveTokenIdentity(COLLIDING_A).length <= 7, true);
});

test("the hash is stable across the shapes a credential arrives in", () => {
  // Whitespace and case are meaningful in a credential, so they must be meaningful
  // here: a trimmed and an untrimmed paste resolve different files on the backend.
  assert.notEqual(resolveTokenIdentity("hf_abc"), resolveTokenIdentity("hf_abc "));
  assert.notEqual(resolveTokenIdentity("hf_abc"), resolveTokenIdentity("HF_ABC"));
});

test("a long or non-ASCII credential still hashes without throwing", () => {
  assert.doesNotThrow(() => resolveTokenIdentity("x".repeat(100_000)));
  assert.doesNotThrow(() => resolveTokenIdentity("héllo-\u{1F600}-token"));
  assert.equal(typeof resolveTokenIdentity("héllo"), "string");
});
