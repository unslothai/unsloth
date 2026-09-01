// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What happens to an explicit Context Length once the load that used it finishes.
//
// The send rule honors a pin in every GPU Memory mode, but the survival rule used to be
// narrower: only a Manual + Auto-layers load kept its pin, so a load on Default sent the
// user's 262,144 and then dropped it. The panel re-seeded to Auto and the next load sent
// 0, which the backend resolves to its host-offload fallback of 8,192 (_AUTO_OFFLOAD_CTX)
// for a model that fits no GPU subset: "it's ignoring my Context Length and always stop
// at Auto (8,192)".
//
// The rule is now "the pin is the Context Length the user EXPLICITLY set, Auto pins
// nothing". Not "the n_ctx the load was invoked with": those are different questions.
// resolveLoadMaxSeqLength sends the resolved context on a same-model reload so the reload
// does not resize, so with the control on Auto a custom or modified preset puts a positive
// n_ctx on the wire, and reading a pin back out of it converted Auto into a numeric pin at
// the current context -- after which a GPU memory change can no longer auto-resize.
//
// So both of these have to hold at once, and they pull against each other:
//   1. an explicitly set context survives a completed load, in EVERY GPU Memory mode;
//   2. Auto stays Auto across a same-model reload, under EVERY preset source.
// The three in-app writers are checked at the source, like the llama-extra-args status test
// next door: each sits inside one large object literal with no seam to call. The status
// path has a seam (resolveCtxPinSeed) and is checked through it.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const HERE = path.dirname(fileURLToPath(import.meta.url));
const read = (relative: string) =>
  readFileSync(path.join(HERE, "..", relative), "utf8");

const APPLIER = read(
  "src/features/chat/lib/apply-inference-status-to-store.ts",
);
const RUNTIME = read("src/features/chat/hooks/use-chat-model-runtime.ts");
const ADAPTER = read("src/features/chat/api/chat-adapter.ts");
const COMPOSER = read("src/features/chat/shared-composer.tsx");
const CONFIG_PAGE = read(
  "src/features/model-picker/components/model-config-page.tsx",
);

const policy = await import("../src/features/chat/presets/preset-policy.ts");
const { resolveCtxPinSeed } = await import(
  "../src/features/chat/lib/resolve-ctx-pin-seed.ts"
);
const { loadedConfigSignature } = await import(
  "../src/features/model-picker/model-config/config-signature.ts"
);

const RETIRED_PREDICATE = /resolveManualAutoCtxPin/;
/** A writer feeding a WIRE value back in as the pin: the regression this guards. */
const WIRE_AS_PIN =
  /resolveExplicitCtxPin\([^)]*\b(?:loadMaxSeqLength|fitMaxSeqLength|compareMaxSeqLength|effectiveMaxSeqLength|requested_context_length)\b/;

/** An empty Context Length pair: the control on Auto with a matching baseline. */
const CLEARED = { customContextLength: null, loadedCustomContextLength: null };

const MODEL = "unsloth/Some-Huge-MoE-GGUF";
const OTHER_MODEL = "unsloth/Something-Else-GGUF";
const VARIANT = "UD-Q4_K_XL";
const REQUESTED = 262144;
const PRESET_SOURCES = ["builtin-default", "custom", "modified"] as const;

/** The sources every writer-level assertion runs over. */
const WRITERS = [
  ["applier", APPLIER],
  ["runtime", RUNTIME],
  ["adapter", ADAPTER],
  ["composer", COMPOSER],
] as const;

type Mode = "auto" | "manual";

/**
 * What a completed load leaves in customContextLength, given the Context Length
 * the user had set for it. The three in-app writers all reduce to this.
 */
const pinAfterLoad = (customContextLength: number | null) =>
  policy.resolveExplicitCtxPin(customContextLength);

/** The n_ctx a load would put on the wire as max_seq_length. */
function sentNCtx(
  customContextLength: number | null,
  {
    mode = "auto",
    gpuLayers = -1,
    residentCtx = 0,
    modelId = MODEL,
    currentCheckpoint = MODEL,
    // The configuration the bug was reported in.
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

/** A per-model config as ModelConfigPage seeds it from the live store. */
function configWithPin(pin: number | null) {
  return {
    customContextLength: pin,
    gpuMemoryMode: "auto",
    gpuLayers: -1,
    nCpuMoe: 0,
  } as never;
}

test("a completed load keeps the context it was invoked with, in every mode", () => {
  // No GPU Memory mode in the signature at all: that is the fix.
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
  // The reported failure: the SECOND load came back at 8,192, because the pin
  // was gone by then and 0 went out instead.
  const pin = pinAfterLoad(REQUESTED);
  assert.equal(
    sentNCtx(pin, { residentCtx: REQUESTED }),
    REQUESTED,
    "the reload reverted to Auto",
  );
  // And it stays put over any number of reloads.
  assert.equal(pinAfterLoad(pin), REQUESTED);
});

test("the settings panel does not remount back to Auto", () => {
  // customContextLength is the first field of modelConfigInstanceKey, so a pin
  // that vanishes on completion re-keys the editor and snaps it back to Auto.
  const pin = pinAfterLoad(REQUESTED);
  assert.equal(
    loadedConfigSignature(configWithPin(pin)),
    loadedConfigSignature(configWithPin(REQUESTED)),
    "the editor would remount and lose the pin",
  );
  // ...and the panel would have read the cleared config as Auto.
  assert.notEqual(
    loadedConfigSignature(configWithPin(null)),
    loadedConfigSignature(configWithPin(REQUESTED)),
  );
  assert.notEqual(pin, null);
  // The link this rests on: null is what the panel calls Auto.
  assert.match(
    CONFIG_PAGE,
    /const contextIsAuto = config\.customContextLength == null;/,
  );
});

test("an Auto load still clears the pin and stays Auto", () => {
  // 0 is the wire value for Auto: the backend sizes the context itself, so there
  // is nothing to pin and the control must keep reading Auto.
  const sentAuto = sentNCtx(null);
  assert.equal(sentAuto, 0);
  const pinAfterAutoLoad = pinAfterLoad(null);
  assert.equal(pinAfterAutoLoad, null);
  assert.equal(sentNCtx(pinAfterAutoLoad, { residentCtx: 8192 }), 0);
  // A pin cleared by hand (the user dragging back to Auto) reads the same way.
  assert.equal(policy.resolveExplicitCtxPin(0), null);
  assert.equal(policy.resolveExplicitCtxPin(null), null);
  assert.equal(policy.resolveExplicitCtxPin(undefined), null);
});

test("a model change does not carry the old pin across", () => {
  // Nothing the outgoing model left in the store may reach the incoming one, and
  // the status echo cannot stand in for it: a length that happens to match the
  // old pin is exactly what an Auto load on the new model reports too.
  const seed = (over: Partial<Parameters<typeof resolveCtxPinSeed>[0]> = {}) =>
    resolveCtxPinSeed({
      incoming: REQUESTED,
      isGguf: true,
      seedLoadParams: true,
      modelChanged: true,
      remembered: null,
      ...over,
    });
  // The new model came up on Auto. 0 proves its own meaning, so the pin clears.
  assert.deepEqual(seed({ incoming: 0 }), CLEARED);
  // The new model came up at the outgoing pin's length with nothing saved for it:
  // still cleared, because nothing recorded that a human asked for it.
  assert.deepEqual(seed(), CLEARED);
  // The new model's OWN saved Context Length, corroborated by the server running
  // it, is the only thing that can seed a pin here.
  assert.deepEqual(seed({ remembered: REQUESTED }), {
    customContextLength: REQUESTED,
    loadedCustomContextLength: REQUESTED,
  });
  // A saved pin the running server is NOT honouring seeds nothing.
  assert.deepEqual(seed({ remembered: 8192 }), CLEARED);
  // A non-GGUF has no n_ctx, so a GGUF pin cannot be left standing over it.
  assert.deepEqual(seed({ isGguf: false }), CLEARED);
  // The next load for the new model is Auto, not the old model's length.
  assert.equal(
    sentNCtx(null, { modelId: OTHER_MODEL, currentCheckpoint: MODEL }),
    0,
  );
  // performLoad clears it outright on a cross-model switch, before the load.
  assert.match(RUNTIME, /customContextLength: null,\s*\n\s*\}\);/);
  // The applier reads the saved value through the same resolver the batch sizes
  // use, so "remembered" is this model's record and not the store's leftovers.
  // savedContextPin, not the raw field: a record written before the MLX pin moved
  // still carries it in maxSeqLength.
  assert.match(
    APPLIER,
    /remembered: remembered\?\.remembered \? savedContextPin\(remembered\.config\) : null,/,
  );
});

test("a poll landing mid-load cannot plant the outgoing model's context", () => {
  // The one window where status still answers for the model on its way out.
  // Harmless while the pin was almost always null; now it is a real number.
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
  // And it cannot clear one either: performLoad owns the pair for the whole load.
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
  // An empty seed writes no keys at all, so the store keeps what it had.
  assert.match(APPLIER, /const ctxPinFields = resolveCtxPinSeed\(\{/);
  assert.match(APPLIER, /\.\.\.ctxPinFields,/);
  // A baseline this status will not apply is not a change worth preserving edits over.
  assert.match(
    APPLIER,
    /\(ctxPinFields\.loadedCustomContextLength !== undefined &&\s*\n\s*prevState\.loadedCustomContextLength !==\s*\n?\s*ctxPinFields\.loadedCustomContextLength\);/,
  );
});

test("Auto stays Auto across a same-model reload, under every preset source", () => {
  // resolveLoadMaxSeqLength's isReloadingCurrentGguf branch: outside
  // builtin-default, a same-model reload on Auto puts the RESOLVED context on the
  // wire so the reload does not resize. Reading a pin back out of that turned
  // Auto into a numeric pin at the current context on any same-model reload --
  // applying an unrelated model setting was enough -- after which a GPU memory
  // change could no longer auto-resize the context.
  assert.equal(policy.getPresetSource("Default"), "builtin-default");
  assert.equal(policy.getPresetSource("My Preset"), "custom");
  const onTheWire = Object.fromEntries(
    PRESET_SOURCES.map((presetSource) => [
      presetSource,
      sentNCtx(null, { presetSource, residentCtx: REQUESTED }),
    ]),
  );
  // Not hypothetical: two of the three really do send a number for Auto.
  assert.deepEqual(onTheWire, {
    "builtin-default": 0,
    custom: REQUESTED,
    modified: REQUESTED,
  });
  for (const presetSource of PRESET_SOURCES) {
    // The control was on Auto, so the load leaves it on Auto whatever it sent...
    const pin = pinAfterLoad(null);
    assert.equal(pin, null, `${presetSource} turned Auto into a pin`);
    // ...and the reload after it is still Auto, so the context can still resize.
    assert.equal(
      sentNCtx(pin, { presetSource, residentCtx: REQUESTED }),
      onTheWire[presetSource],
    );
  }
  // The status echo of those reloads says REQUESTED, and is refused as evidence.
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
  // No writer may go back to feeding a wire value in as the pin.
  for (const [label, source] of WRITERS) {
    assert.doesNotMatch(source, WIRE_AS_PIN, `${label} pins a wire value`);
  }
});

test("the clamp that stops a manual reload resizing is not a user pin", () => {
  // performLoad substitutes the resolved context for Auto when layers are pinned
  // on the same model, or the reload would send 0 and llama.cpp's --fit-off
  // branch would take that as the NATIVE context. That is the app protecting the
  // load, not the user choosing a length, so the pin is captured BEFORE it.
  const capture = RUNTIME.indexOf("const explicitCtxPin = loadCustomContextLength;");
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
  // The regression: all three took the resolved n_ctx, which is the user's number
  // only when there is one, and the current context otherwise.
  assert.match(
    RUNTIME,
    /const keepCustomCtx = resolveExplicitCtxPin\(\s*\n\s*loadResponse\.is_gguf \|\|\s*\n\s*isServedByMlx\([^)]*\)\s*\n\s*\? explicitCtxPin\s*\n\s*: null,\s*\n\s*\);/,
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
  // Each argument is that path's own per-model setting, which is also what fed
  // its send rule, so a pinned load's -c and its pin cannot disagree.
  assert.match(COMPOSER, /const effectiveCustomContextLength = ownConfig\.customContextLength;/);
  assert.match(
    ADAPTER,
    /customContextLength: config\.customContextLength,\s*\n\s*loadedContextLength: null,/,
  );
});

test("the Manual-only predicate is retired, not left to be picked up again", () => {
  // It answered a different question and had no caller left once the four
  // writers moved off it.
  assert.equal(
    (policy as Record<string, unknown>).resolveManualAutoCtxPin,
    undefined,
  );
  // Neither is the wire-valued predicate that replaced it and read Auto as a pin.
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
  // The SEND rule it was named after stays: Manual + Auto layers still sends 0
  // when there is no pin, because llama.cpp's --fit owns the sizing there.
  assert.equal(sentNCtx(null, { mode: "manual", gpuLayers: -1 }), 0);
  assert.equal(
    sentNCtx(REQUESTED, { mode: "manual", gpuLayers: -1 }),
    REQUESTED,
  );
});

test("another client reloading the same model at a new context invalidates the baseline", () => {
  // The pin here describes an invocation that is no longer running. Keeping it
  // means the next unrelated Apply resends it and silently takes back the other
  // client's context, so a contradicted baseline is dropped. The echo still
  // cannot say whether a human chose the new value, so it is not adopted either.
  const seed = (over: Partial<Parameters<typeof resolveCtxPinSeed>[0]> = {}) =>
    resolveCtxPinSeed({
      incoming: REQUESTED,
      isGguf: true,
      seedLoadParams: true,
      modelChanged: false,
      remembered: null,
      ...over,
    });

  // Contradicted: we recorded 8192, the server reports it is running 262144.
  assert.deepEqual(seed({ loadedPin: 8192 }), CLEARED);
  // Agreeing echoes leave the control alone, which is what keeps an explicit pin
  // alive across every ordinary poll.
  assert.deepEqual(seed({ loadedPin: REQUESTED }), {});
  // Auto here holds: with no pin recorded there is no baseline to contradict, and
  // an Auto reload under a custom preset reports a positive n_ctx exactly like
  // this. Adopting it would be the bug this file exists to prevent.
  assert.deepEqual(seed({ loadedPin: null }), {});
  // Our own pinned model reloaded on Auto reports the resolved context, which IS
  // the pin, so it matches and nothing moves.
  assert.deepEqual(seed({ loadedPin: REQUESTED, incoming: REQUESTED }), {});
  // Still nothing mid-load: performLoad owns the pair there.
  assert.deepEqual(seed({ loadedPin: 8192, seedLoadParams: false }), {});
});

test("a positive echo under manual memory with auto layers is an explicit pin", () => {
  // resolveFitMaxSeqLength is what that placement sends, and it answers
  // `customContextLength > 0 ? it : 0`, so Auto is 0 on the wire without
  // exception and a positive echo cannot have come from Auto.
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

  // A fresh tab hydrating a resident pinned server, with nothing saved for it.
  assert.deepEqual(seed(), PINNED);
  // Same on a model change: the echo describes the model that ARRIVED, so it
  // beats both the store's leftovers and a saved config.
  assert.deepEqual(seed({ modelChanged: true }), PINNED);
  // Auto in this mode still reports 0, and 0 still clears.
  assert.deepEqual(seed({ incoming: 0 }), CLEARED);
  // Pinned layers is a different placement: it does not send through
  // resolveFitMaxSeqLength, so the echo is ambiguous again and is not adopted.
  assert.deepEqual(seed({ gpuLayers: 20 }), {});
  // Neither is Default memory, which is where the original bug lived.
  assert.deepEqual(seed({ gpuMemoryMode: "auto", gpuLayers: -1 }), {});
  // The mid-load window outranks it, as it does everything else here.
  assert.deepEqual(seed({ seedLoadParams: false }), {});
  // The applier passes the RAW status fields: the normalised ones null out
  // layers off manual, which would read as "not reported" rather than as Auto.
  assert.match(APPLIER, /gpuMemoryMode: status\.gpu_memory_mode \?\? null,/);
  assert.match(APPLIER, /gpuLayers: status\.gpu_layers \?\? null,/);
  assert.match(
    APPLIER,
    /loadedPin: prevState\.loadedCustomContextLength \?\? null,/,
  );
});

test("a resident MLX pin from another tab is adopted, not read as Auto", () => {
  // An unpinned MLX load sends 0, so a positive echo can only be an explicit pin.
  // Model changed underneath this tab and there is no saved config to corroborate it.
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

  // Auto still clears: 0 is the wire form and proves its own meaning.
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

  // GGUF keeps the old rule: its positive echo is the ambiguous resolved n_ctx.
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
