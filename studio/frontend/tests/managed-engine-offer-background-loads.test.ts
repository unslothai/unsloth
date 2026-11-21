// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Auto-load skips, and compare panes offer, what managed_engine_offer flags, like the picker (#13135, #8861).

import assert from "node:assert/strict";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";
import { DEFAULT_ENGINE_LAUNCH_FIELDS } from "../src/features/chat/lib/engine-launch-fields.ts";

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
const AsyncFunction = Object.getPrototypeOf(async () => {}).constructor;
const paneDecisionBody = new AsyncFunction(
  "paneEngine",
  "validation",
  "deps",
  `const { sel, compareModelDisplayName, confirmManagedEngineIfNeeded, validateModel, panePayload,
     throwIfCompareCancelled, compareSignal, ownConfig, reconcilePersistedGpuIds,
     DEFAULT_ENGINE_LAUNCH_FIELDS } = deps;
   let paneEngineFields = { engine: paneEngine };
   ${composer.slice(paneStart, paneEnd)}
   return paneEngineFields;`,
);
const sel = { id: "unsloth/Qwen3.8-27B-NVFP4" };

/** Runs the pane's offer step; `choice` is what the dialog answers. */
async function paneDecision(
  paneEngine: string,
  validation: Record<string, unknown>,
  choice: string | null = null,
  ownConfig: Record<string, unknown> = {},
) {
  const asked: unknown[] = [];
  let revalidated = 0;
  const fields = await paneDecisionBody(paneEngine, validation, {
    sel,
    compareModelDisplayName: (id: string) => id.split("/").pop() ?? id,
    confirmManagedEngineIfNeeded: async (model: string, offer: unknown) => {
      asked.push([model, offer]);
      return choice;
    },
    panePayload: () => ({}),
    validateModel: async () => {
      revalidated += 1;
      return { managed_engine_offer: null, revalidated: true };
    },
    throwIfCompareCancelled: () => {},
    compareSignal: undefined,
    ownConfig,
    reconcilePersistedGpuIds: (ids: number[] | null) => ids,
    DEFAULT_ENGINE_LAUNCH_FIELDS,
  });
  return { fields, asked, revalidated, validation };
}

test("a default-engine compare pane offers vLLM and loads with the pick", async () => {
  const run = await paneDecision(
    "auto",
    { managed_engine_offer: OFFER },
    "vllm",
  );
  assert.deepEqual(run.asked, [[sel.id, OFFER]]);
  assert.deepEqual(run.fields, {
    engine: "vllm",
    engine_precision: "auto",
    engine_parallelism: "tensor",
    ...DEFAULT_ENGINE_LAUNCH_FIELDS,
    load_in_4bit: false,
  });
  assert.equal(
    run.revalidated,
    1,
    "re-validated as the chosen engine before /load",
  );
  assert.equal(run.validation.revalidated, true);
});

test("the pane's saved GPU pick survives the switch to the offered engine", async () => {
  const run = await paneDecision(
    "auto",
    { managed_engine_offer: OFFER },
    "vllm",
    { selectedGpuIds: [1, 2], selectedGpuIndexKind: "physical" },
  );
  assert.deepEqual(run.fields.gpu_ids, [1, 2]);
});

test("declining the offer refuses the load and names the engine", async () => {
  await assert.rejects(
    paneDecision("auto", { managed_engine_offer: OFFER }, null),
    /Qwen3\.8-27B-NVFP4 is quantized with compressed-tensors, which the default engine cannot run\. Set Inference engine to vLLM/,
  );
  await assert.rejects(
    paneDecision(
      "auto",
      { managed_engine_offer: { quantization: "awq", engines: ["sglang"] } },
      null,
    ),
    /Set Inference engine to SGLang/,
  );
});

test("a pane already on vLLM or SGLang, or an ordinary checkpoint, is not asked", async () => {
  for (const [engine, validation] of [
    ["vllm", { managed_engine_offer: OFFER }],
    ["sglang", { managed_engine_offer: OFFER }],
    ["auto", { managed_engine_offer: null }],
    ["auto", {}],
  ] as const) {
    const run = await paneDecision(engine, { ...validation });
    assert.deepEqual(run.asked, []);
    assert.equal(run.revalidated, 0);
    assert.deepEqual(run.fields, { engine });
  }
});
