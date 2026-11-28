// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import ts from "@typescript/typescript6";
import { readSrc } from "./helpers/kit.ts";
import { enhanceRestoreFor } from "../src/features/images/lib/prompt-enhance.ts";
import { restoredInpaintMaskOptions } from "../src/features/images/lib/inpaint-options.ts";
import { detailerNotApplied, restoredDetailer } from "../src/features/images/lib/detailer.ts";
import { fastCfgEndFromRecipe, fastCfgNotApplied } from "../src/features/images/lib/fast-cfg.ts";
import * as workflows from "../src/features/images/workflows.ts";
import { isWorkflowEnabled } from "../src/features/images/stores/image-workflow-store.ts";

function restoreSettingsWith(negativeCapable: boolean, supported: string[] | null = null) {
  const source = readSrc("features/images/images-page.tsx");
  const tree = ts.createSourceFile(
    "images-page.tsx",
    source,
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TSX,
  );
  let declaration = "";
  const visit = (node: ts.Node) => {
    if (
      ts.isVariableDeclaration(node) &&
      node.name.getText(tree) === "restoreSettings"
    )
      declaration = `const ${node.getText(tree)};`;
    ts.forEachChild(node, visit);
  };
  visit(tree);
  assert.ok(declaration);
  const { outputText } = ts.transpileModule(
    `${declaration}\nreturn restoreSettings;`,
    {
      compilerOptions: { target: ts.ScriptTarget.ES2022 },
    },
  );
  const state: Record<string, unknown> = {};
  const known: Record<string, unknown> = {
    useCallback: (fn: unknown) => fn,
    negativeCapable,
    sizeLimits: { maxSide: 2048 },
    MIN_DIM: 256,
    restorableSize: (width: number, height: number) => ({ width, height }),
    matchAspect: () => ({ key: "1:1", portrait: false }),
    restoreInputsNote: () => null,
    CONDITIONED_WORKFLOW_INPUTS: {},
    DEFAULT_SAMPLER: "default",
    AUTOMATIC_SCHEDULE: "automatic",
    samplerOptions: [],
    changeSamplerChoice: () => {},
    // Its return value is read, so it cannot be a recording no-op.
    enhanceRestoreFor,
    restoredInpaintMaskOptions,
    restoredDetailer,
    detailerNotApplied,
    fastCfgEndFromRecipe,
    fastCfgNotApplied,
    recipeWorkflowTab: workflows.recipeWorkflowTab,
    recipeWorkflowLabel: workflows.recipeWorkflowLabel,
    isWorkflowEnabled,
    supported,
    toast: {
      success: (title: string) => (state.toastSuccess = title),
      warning: (title: string) => (state.toastWarning = title),
    },
  };
  // Every closure name the callback reads that is not a real global resolves to a recording no-op,
  // so a new dependency of restoreSettings (#13569 added changeSamplerChoice / samplerOptions) does
  // not throw a ReferenceError before the negative prompt is reached. Globals (String, Number, Array)
  // still resolve to the real ones.
  const scope = new Proxy(known, {
    has: (target, key) =>
      typeof key === "string" && (key in target || !(key in globalThis)),
    get: (target, key) =>
      typeof key === "string" && key in target
        ? target[key]
        : (...args: unknown[]) => {
            state[key as string] = args.length > 1 ? args : args[0];
          },
  });
  const restore = new Function("scope", `with (scope) { ${outputText} }`)(
    scope,
  );
  return { restore, state };
}

test("restoring an image keeps its negative prompt while a model that ignores one is loaded", () => {
  const { restore, state } = restoreSettingsWith(false);
  restore({
    prompt: "a lighthouse",
    negative_prompt: "blurry, text",
    guidance: 5,
    steps: 30,
    seed: 7,
    width: 1024,
    height: 1024,
    workflow: "create",
  });
  assert.equal(state.setNegativePrompt, "blurry, text");
});

test("restoring an enhanced FLUX.3 image sets the Enhance prompt toggle from its outcome", () => {
  const image = {
    prompt: "a cat",
    enhanced_prompt: "A ginger cat on a windowsill.",
    prompt_enhancement: "applied",
    guidance: 1,
    steps: 16,
    seed: 7,
    width: 1024,
    height: 1024,
    workflow: "create",
  };
  const asTyped = restoreSettingsWith(true);
  asTyped.restore(image);
  assert.deepEqual(asTyped.state.setPromptFor, ["create", "a cat"]);
  assert.equal(asTyped.state.setEnhanceOverride, true);

  const exact = restoreSettingsWith(true);
  exact.restore(image, true);
  assert.deepEqual(exact.state.setPromptFor, ["create", "A ginger cat on a windowsill."]);
  assert.equal(exact.state.setEnhanceOverride, false);

  const legacy = restoreSettingsWith(true);
  legacy.restore({ ...image, enhanced_prompt: undefined, prompt_enhancement: undefined });
  assert.equal("setEnhanceOverride" in legacy.state, false);
});

test("restoring an image-input recipe reopens its own tab and leaves the Create draft alone", () => {
  const cases: Array<[string, string]> = [
    ["img2img", "transform"],
    ["inpaint", "inpaint"],
    ["outpaint", "extend"],
    ["upscale", "upscale"],
    ["edit", "edit"],
    ["reference", "reference"],
    ["txt2img", "create"],
    ["controlnet", "create"],
  ];
  for (const [workflow, tab] of cases) {
    const { restore, state } = restoreSettingsWith(true);
    restore({ prompt: "a fox", guidance: 5, steps: 20, seed: 7, width: 1024, height: 1024, workflow, strength: 0.4 });
    assert.equal(state.setWorkflow, tab, workflow);
    assert.deepEqual(state.setPromptFor, [tab, "a fox"], workflow);
  }
});

test("restoring a recipe whose tab the loaded model lacks applies nothing and warns", () => {
  const image = { prompt: "a fox", guidance: 5, steps: 20, seed: 7, width: 1024, height: 1024, workflow: "inpaint" };
  const lacking = restoreSettingsWith(true, ["create", "edit"]);
  assert.equal(lacking.restore(image), false);
  assert.match(String(lacking.state.toastWarning), /Inpaint, which the loaded model cannot run/);
  for (const key of ["setWorkflow", "setPromptFor", "setSeed", "setSteps", "toastSuccess"])
    assert.equal(key in lacking.state, false, key);
  const capable = restoreSettingsWith(true, ["create", "inpaint"]);
  assert.equal(capable.restore(image), true);
  assert.equal(capable.state.setWorkflow, "inpaint");
  assert.ok(capable.state.toastSuccess);
});
