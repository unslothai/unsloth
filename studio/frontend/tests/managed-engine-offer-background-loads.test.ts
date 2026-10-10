// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Auto-load and compare panes must honour managed_engine_offer like the picker (#13135, #8861).

import assert from "node:assert/strict";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";

const OFFER = {
  quantization: "compressed-tensors",
  engines: ["vllm", "sglang"],
};

const adapter = readSrc("features/chat/api/chat-adapter.ts");
const guardStart = adapter.indexOf("async function canAutoLoad(");
const decisionStart = adapter.indexOf(
  "if (\n      validation.requires_trust_remote_code",
  guardStart,
);
const decisionEnd = adapter.indexOf("return true;\n  }", decisionStart);
assert.ok(
  guardStart >= 0 && decisionStart > guardStart && decisionEnd > decisionStart,
);
const autoLoadDecision = new Function(
  "validation",
  "state",
  `let { blockedByTrustRemoteCode, hadNonTrustFailure, validateFailures } = state;
   const out = (ok) => ({ ok, blockedByTrustRemoteCode, hadNonTrustFailure, validateFailures });
   ${adapter.slice(decisionStart, decisionEnd).replaceAll("return false;", "return out(false);")}
   return out(true);`,
) as (
  validation: Record<string, unknown>,
  state: Record<string, unknown>,
) => { ok: boolean; hadNonTrustFailure: boolean; validateFailures: number };
const fresh = () => ({
  blockedByTrustRemoteCode: false,
  hadNonTrustFailure: false,
  validateFailures: 0,
});

test("auto-load skips a checkpoint the default engine cannot run", () => {
  const result = autoLoadDecision({ managed_engine_offer: OFFER }, fresh());
  assert.equal(result.ok, false);
  assert.equal(
    result.validateFailures,
    1,
    "a skipped candidate spends the validate budget",
  );
  assert.equal(result.hadNonTrustFailure, true);
});

test("auto-load still loads an ordinary checkpoint", () => {
  for (const offer of [undefined, null]) {
    assert.equal(
      autoLoadDecision({ managed_engine_offer: offer }, fresh()).ok,
      true,
    );
  }
});

const composer = readSrc("features/chat/shared-composer.tsx");
const paneStart = composer.indexOf("const engineOffer =");
const paneEnd = composer.indexOf("// Upgrade dialog first", paneStart);
assert.ok(paneStart >= 0 && paneEnd > paneStart);
const paneDecision = new Function(
  "paneEngine",
  "validation",
  "sel",
  "compareModelDisplayName",
  composer.slice(paneStart, paneEnd),
) as (
  paneEngine: string,
  validation: Record<string, unknown>,
  sel: { id: string },
  name: (id: string) => string,
) => void;
const name = (id: string) => id.split("/").pop() ?? id;
const sel = { id: "unsloth/Qwen3.8-27B-NVFP4" };

test("a default-engine compare pane refuses the checkpoint and names the engine", () => {
  assert.throws(
    () => paneDecision("auto", { managed_engine_offer: OFFER }, sel, name),
    /Qwen3\.8-27B-NVFP4 is quantized with compressed-tensors, which the default engine cannot run\. Set Inference engine to vLLM/,
  );
  assert.throws(
    () =>
      paneDecision(
        "auto",
        { managed_engine_offer: { quantization: "awq", engines: ["sglang"] } },
        sel,
        name,
      ),
    /Set Inference engine to SGLang/,
  );
});

test("a compare pane already on vLLM or SGLang, or an ordinary checkpoint, loads", () => {
  paneDecision("vllm", { managed_engine_offer: OFFER }, sel, name);
  paneDecision("sglang", { managed_engine_offer: OFFER }, sel, name);
  paneDecision("auto", { managed_engine_offer: null }, sel, name);
  paneDecision("auto", {}, sel, name);
});
