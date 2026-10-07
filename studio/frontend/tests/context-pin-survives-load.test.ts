// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// An explicitly set Context Length survives a load in every GPU Memory mode, while Auto stays
// Auto across same-model reloads even when a positive n_ctx goes on the wire.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const APPLIER = readSrc("features/chat/lib/apply-inference-status-to-store.ts");
const RUNTIME = readSrc("features/chat/hooks/use-chat-model-runtime.ts");
const ADAPTER = readSrc("features/chat/api/chat-adapter.ts");
const COMPOSER = readSrc("features/chat/shared-composer.tsx");
const CONFIG_PAGE = readSrc("features/model-picker/components/model-config-page.tsx");

const policy = await import("../src/features/chat/presets/preset-policy.ts");
const { resolveCtxPinSeed } = await import(
  "../src/features/chat/lib/resolve-ctx-pin-seed.ts"
);
const { loadedConfigSignature } = await import(
  "../src/features/model-picker/model-config/config-signature.ts"
);

const RETIRED_PREDICATE = /resolveManualAutoCtxPin/;
const WIRE_AS_PIN =
  /resolveExplicitCtxPin\([^)]*\b(?:loadMaxSeqLength|fitMaxSeqLength|compareMaxSeqLength|effectiveMaxSeqLength|requested_context_length)\b/;

const CLEARED = { customContextLength: null, loadedCustomContextLength: null };

const MODEL = "unsloth/Some-Huge-MoE-GGUF";
const OTHER_MODEL = "unsloth/Something-Else-GGUF";
const VARIANT = "UD-Q4_K_XL";
const REQUESTED = 262144;
const PRESET_SOURCES = ["builtin-default", "custom", "modified"] as const;

const WRITERS = [
  ["applier", APPLIER],
  ["runtime", RUNTIME],
  ["adapter", ADAPTER],
  ["composer", COMPOSER],
] as const;

type Mode = "auto" | "manual";

const pinAfterLoad = (customContextLength: number | null) =>
  policy.resolveExplicitCtxPin(customContextLength);

function sentNCtx(
  customContextLength: number | null,
  {
    mode = "auto",
    gpuLayers = -1,
    residentCtx = 0,
    modelId = MODEL,
    currentCheckpoint = MODEL,
    presetSource = "builtin-default",
  }: {
    mode?: Mode;
    gpuLayers?: number;
    residentCtx?: number;
    modelId?: string;
    currentCheckpoint?: string;
    presetSource?: (typeof PRESET_SOURCES)[number];
  } = {},
): number {
  return policy.resolveFitMaxSeqLength(
    true,
    mode,
    gpuLayers,
    customContextLength,
    policy.resolveLoadMaxSeqLength({
      modelId,
      ggufVariant: VARIANT,
      isGguf: true,
      customContextLength,
      loadedContextLength: residentCtx,
      currentCheckpoint,
      activeGgufVariant: VARIANT,
      pinnedMaxSeqLength: null,
      defaultMaxSeqLength: 4096,
      presetSource,
    }),
  );
}

function configWithPin(pin: number | null) {
  return {
    customContextLength: pin,
    gpuMemoryMode: "auto",
    gpuLayers: -1,
    nCpuMoe: 0,
  } as never;
}

test("a completed load keeps the context it was invoked with, in every mode", () => {
  for (const [mode, gpuLayers] of [
    ["auto", -1],
    ["auto", 20],
    ["manual", -1],
    ["manual", 20],
  ] as const) {
    const sent = sentNCtx(REQUESTED, { mode, gpuLayers });
    assert.equal(sent, REQUESTED, `${mode}/${gpuLayers} sent the wrong n_ctx`);
    assert.equal(
      pinAfterLoad(REQUESTED),
      REQUESTED,
      `${mode}/${gpuLayers} dropped the pin`,
    );
  }
});

test("the next load still sends the user's number", () => {
  const pin = pinAfterLoad(REQUESTED);
  assert.equal(
    sentNCtx(pin, { residentCtx: REQUESTED }),
    REQUESTED,
    "the reload reverted to Auto",
  );
  assert.equal(pinAfterLoad(pin), REQUESTED);
});

test("the settings panel does not remount back to Auto", () => {
  const pin = pinAfterLoad(REQUESTED);
  assert.equal(
    loadedConfigSignature(configWithPin(pin)),
    loadedConfigSignature(configWithPin(REQUESTED)),
    "the editor would remount and lose the pin",
  );
  assert.notEqual(
    loadedConfigSignature(configWithPin(null)),
    loadedConfigSignature(configWithPin(REQUESTED)),
  );
  assert.notEqual(pin, null);
  assert.match(
    CONFIG_PAGE,
    /const contextIsAuto = config\.customContextLength == null;/,
  );
});

test("an Auto load still clears the pin and stays Auto", () => {
  const sentAuto = sentNCtx(null);
  assert.equal(sentAuto, 0);
  const pinAfterAutoLoad = pinAfterLoad(null);
  assert.equal(pinAfterAutoLoad, null);
  assert.equal(sentNCtx(pinAfterAutoLoad, { residentCtx: 8192 }), 0);
  assert.equal(policy.resolveExplicitCtxPin(0), null);
  assert.equal(policy.resolveExplicitCtxPin(null), null);
  assert.equal(policy.resolveExplicitCtxPin(undefined), null);
});

test("a model change does not carry the old pin across", () => {
  const seed = (over: Partial<Parameters<typeof resolveCtxPinSeed>[0]> = {}) =>
    resolveCtxPinSeed({
      incoming: REQUESTED,
      isGguf: true,
      seedLoadParams: true,
      modelChanged: true,
      remembered: null,
      ...over,
    });
  assert.deepEqual(seed({ incoming: 0 }), CLEARED);
  assert.deepEqual(seed(), CLEARED);
  assert.deepEqual(seed({ remembered: REQUESTED }), {
    customContextLength: REQUESTED,
    loadedCustomContextLength: REQUESTED,
  });
  assert.deepEqual(seed({ remembered: 8192 }), CLEARED);
  assert.deepEqual(seed({ isGguf: false }), CLEARED);
  assert.equal(
    sentNCtx(null, { modelId: OTHER_MODEL, currentCheckpoint: MODEL }),
    0,
  );
  assert.match(RUNTIME, /customContextLength: null,\s*\n\s*\}\);/);
  assert.match(
    APPLIER,
    /remembered: remembered\?\.remembered \? savedContextPin\(remembered\.config\) : null,/,
  );
});

test("a poll landing mid-load cannot plant the outgoing model's context", () => {
  assert.deepEqual(
    resolveCtxPinSeed({
      incoming: REQUESTED,
      isGguf: true,
      seedLoadParams: false,
      modelChanged: true,
      remembered: REQUESTED,
    }),
    {},
    "a mid-load poll planted a pin",
  );
  assert.deepEqual(
    resolveCtxPinSeed({
      incoming: 0,
      isGguf: true,
      seedLoadParams: false,
      modelChanged: false,
      remembered: null,
    }),
    {},
    "a mid-load poll cleared the pin",
  );
  assert.match(APPLIER, /const ctxPinFields = resolveCtxPinSeed\(\{/);
  assert.match(APPLIER, /\.\.\.ctxPinFields,/);
  assert.match(
    APPLIER,
    /\(ctxPinFields\.loadedCustomContextLength !== undefined &&\s*\n\s*prevState\.loadedCustomContextLength !==\s*\n?\s*ctxPinFields\.loadedCustomContextLength\);/,
  );
});

test("Auto stays Auto across a same-model reload, under every preset source", () => {
  // Same-model Auto reloads outside builtin-default send the resolved context; it is not a pin.
  assert.equal(policy.getPresetSource("Default"), "builtin-default");
  assert.equal(policy.getPresetSource("My Preset"), "custom");
  const onTheWire = Object.fromEntries(
    PRESET_SOURCES.map((presetSource) => [
      presetSource,
      sentNCtx(null, { presetSource, residentCtx: REQUESTED }),
    ]),
  );
  assert.deepEqual(onTheWire, {
    "builtin-default": 0,
    custom: REQUESTED,
    modified: REQUESTED,
  });
  for (const presetSource of PRESET_SOURCES) {
    const pin = pinAfterLoad(null);
    assert.equal(pin, null, `${presetSource} turned Auto into a pin`);
    assert.equal(
      sentNCtx(pin, { presetSource, residentCtx: REQUESTED }),
      onTheWire[presetSource],
    );
  }
  assert.deepEqual(
    resolveCtxPinSeed({
      incoming: REQUESTED,
      isGguf: true,
      seedLoadParams: true,
      modelChanged: false,
      remembered: REQUESTED,
    }),
    {},
    "the status path re-pinned an Auto reload",
  );
  for (const [label, source] of WRITERS) {
    assert.doesNotMatch(source, WIRE_AS_PIN, `${label} pins a wire value`);
  }
});

test("the clamp that stops a manual reload resizing is not a user pin", () => {
  // The pin is captured before performLoad substitutes the resolved context for Auto.
  const capture = RUNTIME.indexOf("const explicitCtxPin = loadRequestContextPin(");
  const clamp = RUNTIME.indexOf("loadCustomContextLength = loadContextLength;");
  assert.notEqual(capture, -1, "the load no longer captures the user's setting");
  assert.notEqual(clamp, -1);
  assert.ok(
    capture < clamp,
    "the clamp now runs before the pin is captured, so the app's substituted length " +
      "would be recorded as the user's choice",
  );
});

test("the three in-app writers pin what the user asked for, not what they sent", () => {
  assert.match(
    RUNTIME,
    /const keepCustomCtx = resolveExplicitCtxPin\(\s*\n\s*loadResponse\.is_gguf \|\| targetIsMlx \? explicitCtxPin : null,\s*\n\s*\);/,
  );
  assert.match(
    RUNTIME,
    /const explicitCtxPin = loadRequestContextPin\(\s*\n\s*loadCustomContextLength,\s*\n\s*targetIsMlx,\s*\n\s*pinnedMaxSeqLength,\s*\n\s*\);/,
  );
  assert.match(
    RUNTIME,
    /customContextLength: keepCustomCtx,\s*\n\s*loadedCustomContextLength: keepCustomCtx,/,
  );
  assert.match(
    ADAPTER,
    /const keepCustomCtx = resolveExplicitCtxPin\(config\.customContextLength\);/,
  );
  assert.match(ADAPTER, /customContextLength: keepCustomCtx,/);
  assert.match(ADAPTER, /loadedCustomContextLength: keepCustomCtx,/);
  assert.match(
    COMPOSER,
    /const keepCustomCtx = targetIsGguf\s*\n\s*\? resolveExplicitCtxPin\(effectiveCustomContextLength\)\s*\n\s*: retainedContextPin\(\{/,
  );
  assert.match(COMPOSER, /customContextLength: keepCustomCtx,/);
  assert.match(COMPOSER, /loadedCustomContextLength: keepCustomCtx,/);
  assert.match(COMPOSER, /const effectiveCustomContextLength = ownConfig\.customContextLength;/);
  assert.match(
    ADAPTER,
    /customContextLength: config\.customContextLength,\s*\n\s*loadedContextLength: null,/,
  );
});

test("the Manual-only predicate is retired, not left to be picked up again", () => {
  assert.equal(
    (policy as Record<string, unknown>).resolveManualAutoCtxPin,
    undefined,
  );
  assert.equal(
    (policy as Record<string, unknown>).resolveLoadedCtxPin,
    undefined,
  );
  for (const [label, source] of WRITERS) {
    assert.doesNotMatch(
      source,
      RETIRED_PREDICATE,
      `${label} still calls the retired predicate`,
    );
  }
  assert.equal(sentNCtx(null, { mode: "manual", gpuLayers: -1 }), 0);
  assert.equal(
    sentNCtx(REQUESTED, { mode: "manual", gpuLayers: -1 }),
    REQUESTED,
  );
});

test("another client reloading the same model at a new context invalidates the baseline", () => {
  const seed = (over: Partial<Parameters<typeof resolveCtxPinSeed>[0]> = {}) =>
    resolveCtxPinSeed({
      incoming: REQUESTED,
      isGguf: true,
      seedLoadParams: true,
      modelChanged: false,
      remembered: null,
      ...over,
    });

  assert.deepEqual(seed({ loadedPin: 8192 }), CLEARED);
  assert.deepEqual(seed({ loadedPin: REQUESTED }), {});
  assert.deepEqual(seed({ loadedPin: null }), {});
  assert.deepEqual(seed({ loadedPin: REQUESTED, incoming: REQUESTED }), {});
  assert.deepEqual(seed({ loadedPin: 8192, seedLoadParams: false }), {});
});

test("a positive echo under manual memory with auto layers is an explicit pin", () => {
  assert.equal(policy.resolveFitMaxSeqLength(true, "manual", -1, null, 4096), 0);
  assert.equal(
    policy.resolveFitMaxSeqLength(true, "manual", -1, REQUESTED, 4096),
    REQUESTED,
  );

  const seed = (over: Partial<Parameters<typeof resolveCtxPinSeed>[0]> = {}) =>
    resolveCtxPinSeed({
      incoming: REQUESTED,
      isGguf: true,
      seedLoadParams: true,
      modelChanged: false,
      remembered: null,
      gpuMemoryMode: "manual",
      gpuLayers: -1,
      ...over,
    });
  const PINNED = {
    customContextLength: REQUESTED,
    loadedCustomContextLength: REQUESTED,
  };

  assert.deepEqual(seed(), PINNED);
  assert.deepEqual(seed({ modelChanged: true }), PINNED);
  assert.deepEqual(seed({ incoming: 0 }), CLEARED);
  assert.deepEqual(seed({ gpuLayers: 20 }), {});
  assert.deepEqual(seed({ gpuMemoryMode: "auto", gpuLayers: -1 }), {});
  assert.deepEqual(seed({ seedLoadParams: false }), {});
  assert.match(APPLIER, /gpuMemoryMode: status\.gpu_memory_mode \?\? null,/);
  assert.match(APPLIER, /gpuLayers: status\.gpu_layers \?\? null,/);
  assert.match(
    APPLIER,
    /loadedPin: prevState\.loadedCustomContextLength \?\? null,/,
  );
});

test("a resident MLX pin from another tab is adopted, not read as Auto", () => {
  const seeded = resolveCtxPinSeed({
    incoming: 32768,
    isGguf: true,
    isMlx: true,
    seedLoadParams: true,
    modelChanged: true,
    remembered: null,
    gpuMemoryMode: null,
    gpuLayers: null,
    loadedPin: null,
  });
  assert.equal(seeded.customContextLength, 32768);
  assert.equal(seeded.loadedCustomContextLength, 32768);

  const auto = resolveCtxPinSeed({
    incoming: 0,
    isGguf: true,
    isMlx: true,
    seedLoadParams: true,
    modelChanged: true,
    remembered: null,
    gpuMemoryMode: null,
    gpuLayers: null,
    loadedPin: null,
  });
  assert.equal(auto.customContextLength, null);

  const gguf = resolveCtxPinSeed({
    incoming: 32768,
    isGguf: true,
    isMlx: false,
    seedLoadParams: true,
    modelChanged: true,
    remembered: null,
    gpuMemoryMode: null,
    gpuLayers: null,
    loadedPin: null,
  });
  assert.equal(gguf.customContextLength, null);
});

test("a failed switch rolls back on the backend that served the outgoing model", () => {
  assert.match(
    RUNTIME,
    /const previousIsMlx = residentIsServedByMlx\(\s*\n\s*previousIsGguf,\s*\n\s*platform\.deviceType,\s*\n\s*platform\.chatOnlyReason,\s*\n\s*rollbackState\.loadedIsMlx,\s*\n\s*\);/,
  );
});

test("an NPU load's pin survives hydration: its status echoes the request, null for Auto", () => {
  assert.match(APPLIER, /isGguf:[\s\S]{0,120}status\.is_npu \?\? false/);
  assert.match(APPLIER, /isMlx: \(status\.is_mlx \?\? false\) \|\| \(status\.is_npu \?\? false\)/);
  const seed = (incoming: number | null) =>
    resolveCtxPinSeed({
      incoming,
      isGguf: true,
      isMlx: true,
      seedLoadParams: true,
      modelChanged: false,
      remembered: null,
      gpuMemoryMode: null,
      gpuLayers: null,
      loadedPin: null,
    });
  assert.equal(seed(16384).customContextLength, 16384);
  assert.equal(seed(null).customContextLength, null);
});
