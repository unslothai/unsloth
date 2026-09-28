// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  registerBundlerResolver,
  readSrc,
  installLocalStorageFake,
} from "./helpers/kit.ts";
registerBundlerResolver();
installLocalStorageFake();
const {
  discoverFastPairs,
  resolveFastPairs,
  normalizeFastPairs,
  verifiedFastVariant,
  priceDifference,
} = await import("../src/features/chat/lib/fast-variants.ts");
const pair = { standard: "vendor/model-v2", fast: "vendor/speed-v2" };
const catalog = [
  { id: pair.standard },
  {
    id: pair.fast,
    description: "A faster edition built from the same model-v2 checkpoint.",
  },
];
const detected = discoverFastPairs(catalog);

test("catalog detects an explicit checkpoint claim without model-specific code", () => {
  assert.equal(detected.length, 1);
  assert.equal(detected[0].standard, pair.standard);
  assert.equal(detected[0].fast, pair.fast);
  assert.equal(detected[0].source, "catalog");
  assert.equal(
    verifiedFastVariant(
      "openrouter",
      pair.fast,
      [pair.standard, pair.fast],
      undefined,
      detected,
    )?.isFast,
    true,
  );
  assert.deepEqual(
    discoverFastPairs([{ id: "vendor/model" }, { id: "vendor/model-fast" }]),
    [],
  );
  assert.deepEqual(
    discoverFastPairs([
      ...catalog,
      { id: "vendor/another", description: catalog[1].description },
    ]),
    [],
  );
  assert.deepEqual(
    discoverFastPairs([
      { id: pair.standard },
      {
        id: pair.fast,
        description:
          "A faster edition with different weights, not the same model-v2 checkpoint.",
      },
    ]),
    [],
  );
});

test("user pairs override detection, work for arbitrary identifiers, and can disable discovery", () => {
  const manual = [{ standard: pair.standard, fast: "vendor/custom-turbo" }];
  assert.deepEqual(
    resolveFastPairs(catalog, manual).map((p) => p.fast),
    ["vendor/custom-turbo"],
  );
  assert.deepEqual(resolveFastPairs(catalog, [], false), []);
  assert.equal(
    normalizeFastPairs([
      { standard: "a", fast: "a" },
      { standard: "a", fast: "b" },
      { standard: "b", fast: "c" },
    ]).length,
    1,
  );
});

test("available companions can be enabled directly by the Fast toggle", () => {
  assert.equal(
    verifiedFastVariant(
      "openrouter",
      pair.standard,
      [pair.standard],
      undefined,
      detected,
    )!.reason!,
    null,
  );
  assert.match(
    verifiedFastVariant(
      "openrouter",
      pair.standard,
      [pair.standard, pair.fast],
      [pair.standard],
      detected,
    )!.reason!,
    /no longer available/,
  );
  assert.equal(
    verifiedFastVariant(
      "openrouter",
      pair.standard,
      [pair.standard, pair.fast],
      undefined,
      detected,
    )?.reason,
    null,
  );
  assert.equal(
    verifiedFastVariant(
      "custom",
      pair.standard,
      [pair.fast],
      undefined,
      detected,
    ),
    null,
  );
});

test("price multipliers use independently published rates and require matching input/output ratios", () => {
  const base = { rates: { prompt: "0.000001", completion: "0.000002" } };
  assert.equal(
    priceDifference(base, {
      rates: { prompt: "0.000004", completion: "0.000008" },
    }),
    "4× published rates",
  );
  assert.equal(
    priceDifference(base, {
      rates: { prompt: "0.000004", completion: "0.000006" },
    }),
    "Input 4× · Output 3×",
  );
  assert.equal(priceDifference(base, { rates: { prompt: "0" } }), null);
  assert.equal(
    priceDifference(base, { ...base, overrides: [{ min_prompt_tokens: 10 }] }),
    "Variable rates",
  );
});

test("popover and shortcut share the real model selection action, while native Fast remains a parameter", () => {
  const controls = readSrc("features/chat/lib/fast-controls.ts");
  assert.match(
    controls,
    /buildExternalModelId\(fast.provider.id, fast.variant.destination\)/,
  );
  assert.match(controls, /fast\.busy \|\| !fast\.connected/);
  assert.match(controls, /after\.view\.levels\.includes\(before\.effort\)/);
  assert.match(controls, /fastMode: !before\.state\.params\.fastMode/);
  assert.match(
    readSrc("features/chat/chat-page.tsx"),
    /toggleFast\(handleCheckpointChange\)/,
  );
  assert.match(
    readSrc("features/chat/components/fast-control.tsx"),
    /toggleFast\(/,
  );
});

test("Fast preferences survive save/reload and backend synchronization", async () => {
  const { saveExternalProviders, loadExternalProviders } = await import(
    "../src/features/chat/external-providers.ts"
  );
  const provider = {
    id: "fast-local",
    name: "OpenRouter",
    providerType: "openrouter",
    baseUrl: "https://openrouter.ai/api/v1",
    models: [pair.standard, pair.fast],
    fastPairs: [pair],
    autoDetectFastVariants: false,
    createdAt: 0,
    updatedAt: 0,
  };
  saveExternalProviders([provider]);
  const restored = loadExternalProviders()[0];
  assert.equal(restored.fastPairs?.[0].fast, pair.fast);
  assert.equal(restored.autoDetectFastVariants, false);
  assert.match(
    readSrc("features/chat/sync-external-providers.ts"),
    /fastPairs: providerType === "openrouter" \? existing.fastPairs/,
  );
});

test("native Fast tiers are read from endpoint data and keep their independent prices", async () => {
  const { setOpenRouterFastTier, openRouterFastTier } = await import(
    "../src/features/chat/lib/openrouter-fast-tier.ts"
  );
  const { providerSupportsFastMode } = await import(
    "../src/features/chat/provider-capabilities.ts"
  );
  setOpenRouterFastTier("vendor/future-model", {
    supported: true,
    available: true,
    endpoints: [
      {
        tag: "vendor/fast",
        pricing: { rates: { prompt: "0.00001", completion: "0.00005" } },
      },
    ],
    fetchedAt: Date.now(),
    source: "https://openrouter.ai/api/v1/models/vendor/future-model/endpoints",
  });
  assert.equal(
    providerSupportsFastMode("openrouter", "vendor/future-model"),
    true,
  );
  assert.equal(providerSupportsFastMode("openrouter", "vendor/unknown"), false);
  assert.equal(
    openRouterFastTier("vendor/future-model")?.endpoints[0].pricing?.rates
      .prompt,
    "0.00001",
  );
  assert.equal(providerSupportsFastMode("anthropic", "claude-opus-5"), true);
  assert.equal(providerSupportsFastMode("anthropic", "claude-opus-4-7"), false);
});
