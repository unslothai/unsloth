// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What happens to an explicit Context Length once the load that used it finishes.
//
// The send rule honors a pin in every GPU Memory mode (resolveLoadMaxSeqLength returns
// it before it looks at anything else). The survival rule used to be narrower: only a
// Manual + Auto-layers load kept its pin, so a load on Default sent the user's 262,144
// and then dropped it. customContextLength is the first field of ModelConfigPage's React
// key, so the panel remounted and re-seeded to Auto, and the next load sent 0 -- which
// the backend resolves to its host-offload fallback of 8,192 (_AUTO_OFFLOAD_CTX) for a
// model that fits no GPU subset. The user's report: "it's ignoring my Context Length and
// always stop at Auto (8,192)".
//
// The rule is now "the pin is the n_ctx the load was invoked with, 0 (Auto) clears it".
// The four writers are checked at the source, like the llama-extra-args status test next
// door: each sits inside one large object literal with no seam to call.

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
const { loadedConfigSignature } = await import(
  "../src/features/model-picker/model-config/config-signature.ts"
);

const RETIRED_PREDICATE = /resolveManualAutoCtxPin/;

const MODEL = "unsloth/Some-Huge-MoE-GGUF";
const OTHER_MODEL = "unsloth/Something-Else-GGUF";
const VARIANT = "UD-Q4_K_XL";
const REQUESTED = 262144;

type Mode = "auto" | "manual";

/** The n_ctx a load would put on the wire as max_seq_length. */
function sentNCtx(
  customContextLength: number | null,
  {
    mode = "auto",
    gpuLayers = -1,
    residentCtx = 0,
    modelId = MODEL,
    currentCheckpoint = MODEL,
  }: {
    mode?: Mode;
    gpuLayers?: number;
    residentCtx?: number;
    modelId?: string;
    currentCheckpoint?: string;
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
      ggufContextLength: residentCtx,
      currentCheckpoint,
      activeGgufVariant: VARIANT,
      maxSeqLength: 4096,
      // The store's own default (getPresetSource("Default")), which is the
      // configuration the bug was reported in.
      presetSource: policy.getPresetSource("Default"),
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
  // No GPU Memory mode in the signature at all: that is the fix. The control is
  // offered in every mode, so its pin has to survive in every mode.
  for (const [mode, gpuLayers] of [
    ["auto", -1],
    ["auto", 20],
    ["manual", -1],
    ["manual", 20],
  ] as const) {
    const sent = sentNCtx(REQUESTED, { mode, gpuLayers });
    assert.equal(sent, REQUESTED, `${mode}/${gpuLayers} sent the wrong n_ctx`);
    assert.equal(
      policy.resolveLoadedCtxPin(sent),
      REQUESTED,
      `${mode}/${gpuLayers} dropped the pin`,
    );
  }
});

test("the next load still sends the user's number", () => {
  // The reported failure: the second load is the one that came back at 8,192,
  // because the pin was gone and 0 went out instead.
  const pinAfterLoad = policy.resolveLoadedCtxPin(sentNCtx(REQUESTED));
  assert.equal(
    sentNCtx(pinAfterLoad, { residentCtx: REQUESTED }),
    REQUESTED,
    "the reload reverted to Auto",
  );
  // And it stays put over any number of reloads.
  const pinAfterReload = policy.resolveLoadedCtxPin(
    sentNCtx(pinAfterLoad, { residentCtx: REQUESTED }),
  );
  assert.equal(pinAfterReload, REQUESTED);
});

test("the settings panel does not remount back to Auto", () => {
  // customContextLength is the first field of modelConfigInstanceKey, so a pin
  // that vanishes on completion re-keys the editor, which re-seeds configState
  // from the cleared config and snaps the slider to position 0 (Auto).
  const pinAfterLoad = policy.resolveLoadedCtxPin(sentNCtx(REQUESTED));
  assert.equal(
    loadedConfigSignature(configWithPin(pinAfterLoad)),
    loadedConfigSignature(configWithPin(REQUESTED)),
    "the editor would remount and lose the pin",
  );
  // ...and the panel would have read the cleared config as Auto.
  assert.notEqual(
    loadedConfigSignature(configWithPin(null)),
    loadedConfigSignature(configWithPin(REQUESTED)),
  );
  assert.notEqual(pinAfterLoad, null);
  // The link this rests on: null is what the panel calls Auto.
  assert.match(
    CONFIG_PAGE,
    /const contextIsAuto = config\.customContextLength == null;/,
  );
});

test("an Auto load still clears the pin and stays Auto", () => {
  // 0 is the wire value for Auto; the backend omits -c or sizes the context
  // itself, so there is nothing to pin and the control must keep reading Auto.
  const sentAuto = sentNCtx(null);
  assert.equal(sentAuto, 0);
  const pinAfterAutoLoad = policy.resolveLoadedCtxPin(sentAuto);
  assert.equal(pinAfterAutoLoad, null);
  assert.equal(sentNCtx(pinAfterAutoLoad, { residentCtx: 8192 }), 0);
  // A pin cleared by hand (the user dragging back to Auto) reads the same way.
  assert.equal(policy.resolveLoadedCtxPin(0), null);
  assert.equal(policy.resolveLoadedCtxPin(null), null);
  assert.equal(policy.resolveLoadedCtxPin(undefined), null);
});

test("a model change does not carry the old pin across", () => {
  // The status applier reads requested_context_length, which is the n_ctx of the
  // ACTIVE load, never a previous value held in the store. A different model
  // loaded underneath therefore reports its own, and one that came up on Auto
  // reports 0 -- so the outgoing model's pin is overwritten, not inherited.
  const outgoingPin = policy.resolveLoadedCtxPin(REQUESTED);
  assert.equal(outgoingPin, REQUESTED);
  const incomingOnAuto = { requested_context_length: 0, is_gguf: true };
  assert.equal(
    policy.resolveLoadedCtxPin(incomingOnAuto.requested_context_length),
    null,
  );
  // The next load for the new model is Auto, not the old model's length.
  assert.equal(
    sentNCtx(null, { modelId: OTHER_MODEL, currentCheckpoint: MODEL }),
    0,
  );
  // Sourced from status alone, so nothing in the store can leak across.
  assert.match(
    APPLIER,
    /const loadedCtxPin = status\.is_gguf\s*\n\s*\? resolveLoadedCtxPin\(status\.requested_context_length \?\? null\)\s*\n\s*: null;/,
  );
  // performLoad clears it outright on a cross-model switch, before the load.
  assert.match(RUNTIME, /customContextLength: null,\s*\n\s*\}\);/);
});

test("a poll landing mid-load cannot plant the outgoing model's context", () => {
  // The one window where status still answers for the model on its way out.
  // Harmless while the pin was almost always null; now it is a real number, so
  // the applier takes the same in-flight rule as every other load param.
  assert.match(
    APPLIER,
    /const ctxPinFields = seedLoadParams\s*\n\s*\? \{\s*\n\s*customContextLength: loadedCtxPin,\s*\n\s*loadedCustomContextLength: loadedCtxPin,/,
  );
  assert.match(
    APPLIER,
    /: \{\s*\n\s*customContextLength: prevState\.customContextLength,\s*\n\s*loadedCustomContextLength: prevState\.loadedCustomContextLength,/,
  );
  // A baseline this status will not apply is not a change worth preserving edits over.
  assert.match(
    APPLIER,
    /\(seedLoadParams && prevState\.loadedCustomContextLength !== loadedCtxPin\);/,
  );
});

test("all four writers pin the n_ctx their load actually sent", () => {
  // They disagreed before: two kept the value, two threw it away, and the one
  // the picker uses was the one that threw it away.
  assert.match(
    RUNTIME,
    /const keepCustomCtx = resolveLoadedCtxPin\(\s*\n\s*loadResponse\.is_gguf \? loadMaxSeqLength : null,\s*\n\s*\);/,
  );
  assert.match(
    RUNTIME,
    /customContextLength: keepCustomCtx,\s*\n\s*loadedCustomContextLength: keepCustomCtx,/,
  );
  assert.match(
    ADAPTER,
    /const keepCustomCtx = resolveLoadedCtxPin\(fitMaxSeqLength\);/,
  );
  assert.match(ADAPTER, /customContextLength: keepCustomCtx,/);
  assert.match(ADAPTER, /loadedCustomContextLength: keepCustomCtx,/);
  assert.match(
    COMPOSER,
    /const keepCustomCtx = targetIsGguf\s*\n\s*\? resolveLoadedCtxPin\(compareMaxSeqLength\)\s*\n\s*: null;/,
  );
  assert.match(COMPOSER, /customContextLength: keepCustomCtx,/);
  assert.match(COMPOSER, /loadedCustomContextLength: keepCustomCtx,/);
});

test("the Manual-only predicate is retired, not left to be picked up again", () => {
  // It answered a different question ("does Manual + Auto layers need its pin
  // re-derived") and had no caller left once the four writers moved off it.
  assert.equal(
    (policy as Record<string, unknown>).resolveManualAutoCtxPin,
    undefined,
  );
  for (const [label, source] of [
    ["applier", APPLIER],
    ["runtime", RUNTIME],
    ["adapter", ADAPTER],
    ["composer", COMPOSER],
  ] as const) {
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
