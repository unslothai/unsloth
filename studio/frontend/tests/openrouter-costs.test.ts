// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";
registerBundlerResolver();
installLocalStorageFake();
const { nonnegativeDecimal, formatUsd, tokenRate } = await import(
  "../src/features/chat/lib/model-pricing.ts"
);
const { createCostRecorder, sumCostReceipts, messageCost } = await import(
  "../src/features/chat/lib/cost-receipts.ts"
);
const {
  setProviderModelCatalog,
  exactModelPricing,
  markProviderCatalogRefreshFailed,
} = await import("../src/features/chat/model-catalog.ts");

test("prices preserve free, reject invalid values and display tiny charges", () => {
  for (const invalid of [
    null,
    undefined,
    "",
    " ",
    true,
    "-1",
    -1,
    NaN,
    Infinity,
    "0x10",
    "1USD",
  ])
    assert.equal(nonnegativeDecimal(invalid), null);
  assert.equal(nonnegativeDecimal("1e-8"), 1e-8);
  assert.equal(tokenRate("0"), "$0");
  assert.equal(tokenRate("0.0000025"), "$2.5");
  assert.equal(tokenRate(undefined), "Unavailable");
  assert.notEqual(formatUsd(1e-10), "$0.00");
  assert.equal(formatUsd(0.000001), "$0.000001");
});

test("exact model rates retain conditions, zero and cached status without base fallback", () => {
  const overrides = [{ min_prompt_tokens: 200000, prompt: "0.000004" }];
  setProviderModelCatalog("openrouter", [
    {
      id: "vendor/model",
      pricing: { rates: { prompt: "0", completion: "0.000001" }, overrides },
    },
    {
      id: "vendor/model-fast",
      pricing: { rates: { prompt: "0.00001", completion: "0.00002" } },
    },
  ]);
  assert.equal(exactModelPricing("openrouter", "vendor/model:fast"), null);
  assert.equal(
    exactModelPricing("openrouter", "vendor/model-fast")?.rates.prompt,
    "0.00001",
  );
  assert.deepEqual(
    exactModelPricing("openrouter", "vendor/model")?.overrides,
    overrides,
  );
  markProviderCatalogRefreshFailed("openrouter");
  assert.equal(exactModelPricing("openrouter", "vendor/model")?.cached, true);
});

test("trailing streaming usage is authoritative and duplicate events are idempotent", () => {
  const recorder = createCostRecorder("run", "requested");
  recorder.observe({ _openrouterAttempt: "upstream1" });
  recorder.observe({ id: "g1", model: "served", choices: [{}] });
  assert.equal(sumCostReceipts(recorder.snapshot()).incomplete, true);
  const chunk = {
    id: "g1",
    choices: [],
    usage: {
      cost: 0.00002,
      prompt_tokens: 10,
      cost_details: { upstream_inference_cost: 3 },
    },
  };
  recorder.observe(chunk);
  recorder.observe(chunk);
  const result = sumCostReceipts(recorder.snapshot());
  assert.equal(result.total, 0.00002);
  assert.equal(result.receipts.length, 1);
  assert.equal(result.incomplete, false);
  assert.equal(result.receipts[0].servedModel, "served");
  assert.equal(result.receipts[0].requestedModel, "requested");
  recorder.observe({ id: "g1", usage: { cost: null } });
  assert.equal(sumCostReceipts(recorder.snapshot()).total, 0.00002);
});

test("tool calls, failed attempts, continuations and forks accumulate without shared-generation double counting", () => {
  const first = createCostRecorder("run1", "model");
  first.observe({ _openrouterAttempt: "a1" });
  first.observe({ id: "g1", usage: { cost: 0.1 } });
  first.observe({ _openrouterAttempt: "a2" });
  first.observe({ id: "g2", choices: [{}] });
  const partial = first.snapshot();
  // Final charge arrives after cancellation; keep it, independently of the UI run status.
  first.observe({ id: "g2", usage: { cost: 0.2 } });
  const continued = createCostRecorder(
    "run2",
    "model",
    JSON.parse(JSON.stringify(first.snapshot())),
  );
  continued.observe({ _openrouterAttempt: "a3" }); // failed request without any generation id
  continued.observe({ _openrouterAttempt: "a4" });
  continued.observe({ id: "g4", usage: { cost: 0 } });
  const total = sumCostReceipts([
    ...partial,
    ...first.snapshot(),
    ...continued.snapshot(),
  ]);
  assert.ok(Math.abs(total.total - 0.3) < 1e-10);
  assert.equal(total.incomplete, true);
  assert.equal(total.receipts.length, 4);
  assert.equal(
    messageCost({ responseDetails: { providerType: "openrouter" } }).historical,
    true,
  );
  assert.equal(messageCost({}).relevant, false);
});

test("synthetic tool frames do not create charges; received usage survives reload", () => {
  const recorder = createCostRecorder("run", "model");
  recorder.observe({ _openrouterAttempt: "a" });
  recorder.observe({ id: "synthetic", choices: [{}], ...{ _toolEvent: {} } });
  recorder.observe({ id: "g", choices: [{}] });
  recorder.observe({ usage: { cost: 0.4 } });
  const restored = messageCost({
    costReceipts: JSON.parse(JSON.stringify(recorder.snapshot())),
  });
  assert.equal(restored.total, 0.4);
  assert.equal(restored.receipts.length, 1);
});
