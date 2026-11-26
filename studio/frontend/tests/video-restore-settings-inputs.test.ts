// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import ts from "@typescript/typescript6";
import { clipPrecisionSelection, clipTextEncoderPrecisionSelection } from "../src/lib/clip-precision.ts";
import { readSrc } from "./helpers/kit.ts";

function restoreSettings(extra: Record<string, unknown> = {}) {
  const source = readSrc("features/video/video-page.tsx");
  const tree = ts.createSourceFile("video-page.tsx", source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  const declarations: string[] = [];
  const visit = (node: ts.Node) => {
    if (ts.isVariableDeclaration(node) && /^(restoreSettings|CONDITIONING_INPUTS)$/.test(node.name.getText(tree)))
      declarations.push(`const ${node.getText(tree)};`);
    ts.forEachChild(node, visit);
  };
  visit(tree);
  assert.ok(declarations.length);
  const { outputText } = ts.transpileModule(`${declarations.join("\n")}\nreturn restoreSettings;`, {
    compilerOptions: { target: ts.ScriptTarget.ES2022 },
  });
  const state: Record<string, unknown> = {};
  const toasts: string[] = [];
  const known: Record<string, unknown> = {
    useCallback: (fn: unknown) => fn,
    statusKnown: false,
    sizeLimits: {},
    ASPECT_RATIOS: {},
    resolutionChosen: { current: false },
    fitVideoSize: (width: number, height: number) => [width, height],
    videoSizeAspect: () => ({ key: "16:9", portrait: false }),
    toast: { success: (message: string) => toasts.push(message) },
    ...extra,
  };
  const scope = new Proxy(known, {
    has: (target, key) => typeof key === "string" && (key in target || !(key in globalThis)),
    get: (target, key) =>
      typeof key === "string" && key in target
        ? target[key]
        : (...args: unknown[]) => {
            state[key as string] = args[0];
          },
  });
  const restore = new Function("scope", `with (scope) { ${outputText} }`)(scope);
  return { restore, state, toasts };
}

const recipe = {
  prompt: "a fox in the snow",
  steps: 30,
  guidance: 4,
  seed: 7,
  width: 960,
  height: 544,
  fps: 24,
  num_frames: 121,
};

for (const conditioning of [null, "t2va", "i2va", "l2va", "fl2va", "ref2va", "v2v"]) {
  test(`restoring a ${conditioning ?? "legacy"} recipe drops every staged conditioning input`, () => {
    const { restore, state } = restoreSettings();
    restore({ ...recipe, conditioning });
    assert.equal(state.setFirstFrame, null);
    assert.equal(state.setLastFrame, null);
    assert.equal(state.setContinuationClip, null);
    assert.equal(state.setContinuationKeep, "last");
    assert.deepEqual(state.setReferenceImages, []);
    assert.deepEqual(state.setReferenceVideos, []);
    assert.deepEqual(state.setReferenceAudios, []);
    assert.equal(state.setReferenceImageSize, "match");
    assert.equal(state.setCropPictureIndex, null);
    assert.equal((state.setRestoreGeneration as (n: number) => number)(1), 2);
  });
}

test("restoring a text-only recipe says nothing is missing", () => {
  for (const conditioning of [null, "t2va"]) {
    const { restore, toasts } = restoreSettings();
    restore({ ...recipe, conditioning });
    assert.deepEqual(toasts, ["Settings restored to inputs"]);
  }
});

test("restoring a conditioned recipe names the input to add again", () => {
  const expected: Record<string, RegExp> = {
    i2va: /Add the start frame again/,
    l2va: /Add the end frame again/,
    fl2va: /Add the start and end frames again/,
    ref2va: /Add the references again/,
    v2v: /Add the clip to continue again/,
  };
  for (const [conditioning, message] of Object.entries(expected)) {
    const { restore, toasts } = restoreSettings();
    restore({ ...recipe, conditioning });
    assert.equal(toasts.length, 1);
    assert.match(toasts[0], message);
  }
});

test("restoring a restyle recipe reopens Restyle with its prompt and seed, leaving Create alone", () => {
  const { restore, state, toasts } = restoreSettings();
  restore({ ...recipe, conditioning: "restyle" });
  assert.deepEqual(state.setRestyleRecipe, { prompt: "a fox in the snow", seed: "7" });
  assert.equal(state.setPageMode, "restyle");
  for (const setter of ["setPrompt", "setSteps", "setGuidance", "setResolutionIntent", "setFirstFrame", "setReferenceImages", "setRestoreGeneration"])
    assert.ok(!(setter in state), setter);
  assert.deepEqual(toasts, ["Settings restored. Add the source clip again to reproduce this clip."]);
});

test("restoring a clip restores the text encoder precision it ran with, beside the transformer one", () => {
  const helpers = { clipPrecisionSelection, clipTextEncoderPrecisionSelection };
  for (const [recorded, want] of [
    ["fp8", "fp8"],
    ["fp8_dynamic", "fp8_dynamic"],
    ["int8", "int8"],
    ["nvfp4", "nvfp4"],
    ["off", "none"],
    // the sidecar records the ENGAGED encoder: null beside model_kind is a dense run (a quantized family default
    // records its scheme), so restoring pins dense rather than re-running Auto (LTX-2's Auto picks fp8)
    [null, "none"],
  ] as const) {
    const { restore, state } = restoreSettings(helpers);
    restore({ ...recipe, model_kind: "pipeline", transformer_quant: "int8", text_encoder_quant: recorded });
    assert.equal(state.setTransformerQuant, "int8");
    assert.equal(state.setTextEncoderQuant, want, String(recorded));
  }
  // an old clip with no precision record keeps the current pick
  const { restore, state } = restoreSettings(helpers);
  restore({ ...recipe });
  assert.ok(!("setTextEncoderQuant" in state));
});

test("the video load sends the text encoder precision the select holds", () => {
  const source = readSrc("features/video/video-page.tsx");
  assert.match(source, /label="Text encoder precision"/);
  assert.match(
    source,
    /text_encoder_quant: controls\.textEncoderQuant === "auto" \? undefined : controls\.textEncoderQuant/,
  );
  assert.equal(source.match(/text_encoder_quant: advanced\.text_encoder_quant/g)?.length, 3);
});
