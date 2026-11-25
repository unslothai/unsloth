// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import ts from "@typescript/typescript6";
import { readSrc } from "./helpers/kit.ts";

function restoreSettings() {
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

for (const conditioning of [null, "t2va", "i2va", "l2va", "fl2va", "ref2va", "v2v", "restyle"]) {
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
    restyle: /Add the source clip on the Restyle tab again/,
  };
  for (const [conditioning, message] of Object.entries(expected)) {
    const { restore, toasts } = restoreSettings();
    restore({ ...recipe, conditioning });
    assert.equal(toasts.length, 1);
    assert.match(toasts[0], message);
  }
});
