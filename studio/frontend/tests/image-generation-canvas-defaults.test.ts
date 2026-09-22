// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import ts from "typescript";
import {
  DEFAULT_RESOLUTION,
  canvasSeedFor,
  resolutionFor,
} from "../src/features/images/image-generation-defaults.ts";
import { readSrc } from "./helpers/kit.ts";

test("the backend's recommended canvas is what seeds the size fields", () => {
  assert.deepEqual(resolutionFor({ recommendedCanvas: 512 }), {
    width: 512,
    height: 512,
  });
  assert.deepEqual(resolutionFor({ recommendedCanvas: 1024 }), {
    width: 1024,
    height: 1024,
  });
});

test("no opinion from the backend keeps 1024", () => {
  for (const value of [null, undefined, 0, Number.NaN, -512]) {
    assert.deepEqual(
      resolutionFor({ recommendedCanvas: value }),
      DEFAULT_RESOLUTION,
      String(value),
    );
  }
});

// The page's load poll, run for real: its ready branch is where a pick's canvas has to land, since
// the resident seed effect returns early for any load this page started.
function loadPollReady() {
  const source = readSrc("features/images/images-page.tsx");
  const tree = ts.createSourceFile(
    "images-page.tsx",
    source,
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TSX,
  );
  let declaration = "";
  function visit(node: ts.Node) {
    if (
      ts.isVariableDeclaration(node) &&
      node.name.getText(tree) === "pollLoadProgress"
    )
      declaration = `const ${node.getText(tree)};`;
    ts.forEachChild(node, visit);
  }
  visit(tree);
  assert.ok(declaration, "pollLoadProgress not found");
  const { outputText } = ts.transpileModule(
    `${declaration}\nreturn pollLoadProgress;`,
    { compilerOptions: { target: ts.ScriptTarget.ES2022 } },
  );
  return async (opts: {
    pick: boolean;
    superseded?: boolean;
    current: { width: number; height: number };
    recommended: number | null;
  }) => {
    const seeds: unknown[] = [];
    const scope = {
      useCallback: (fn: unknown) => fn,
      cancelSeq: { current: 0 },
      statusTicket: { current: 0 },
      getDiffusionLoadProgress: async () => ({ phase: "ready" }),
      getDiffusionStatus: async () => ({
        loaded: true,
        repo_id: "Qwen/Qwen-Image-2.1",
        recommended_canvas: opts.recommended,
      }),
      dismissLoadToast: () => {},
      refreshStatus: async () => {},
      cancelLoadFromToast: () => {},
      setStatusIfNewest: () => {},
      toast: { success: () => {} },
      lastLoad: {
        current: { repoId: "Qwen/Qwen-Image-2.1", kind: "pipeline" },
      },
      matchesRememberedModel: () => false,
      rememberImageModel: () => {},
      setRememberedModel: () => {},
      quantRevert: {
        current: opts.pick ? { prev: null, steps: 20, guidance: 4 } : null,
      },
      pickRecipeSuperseded: { current: () => opts.superseded === true },
      canvasSeedFor,
      canvasNow: { current: opts.current },
      seededCanvas: { current: DEFAULT_RESOLUTION },
      seedCanvas: (size: unknown) => seeds.push(size),
      setBusy: () => {},
      setPendingModelDefaults: () => {},
      lastLoadRevert: { current: null },
    };
    const poll = new Function(...Object.keys(scope), outputText)(
      ...Object.values(scope),
    );
    await poll();
    return seeds;
  };
}

test("a model picked on this page gets the backend's canvas when its load lands", async () => {
  const ready = loadPollReady();
  assert.deepEqual(
    await ready({
      pick: true,
      current: { width: 1024, height: 1024 },
      recommended: 512,
    }),
    [{ width: 512, height: 512 }],
  );
});

test("a size the user chose, or a load no pick started, keeps its canvas", async () => {
  const ready = loadPollReady();
  // Typed before the pick or while its download ran.
  assert.deepEqual(
    await ready({
      pick: true,
      current: { width: 768, height: 768 },
      recommended: 512,
    }),
    [],
  );
  // A preset taken after the pick owns the form.
  assert.deepEqual(
    await ready({
      pick: true,
      superseded: true,
      current: { width: 1024, height: 1024 },
      recommended: 512,
    }),
    [],
  );
  // Reapply and a recalled generation reload what the user already set up.
  assert.deepEqual(
    await ready({
      pick: false,
      current: { width: 1024, height: 1024 },
      recommended: 512,
    }),
    [],
  );
});

test("the seed only replaces the page's own last seed", () => {
  const seeded = { width: 512, height: 512 };
  // A roomier model after a tight one restores 1024, since 512 was ours.
  assert.deepEqual(canvasSeedFor(seeded, seeded, 1024), DEFAULT_RESOLUTION);
  assert.deepEqual(canvasSeedFor(seeded, seeded, null), DEFAULT_RESOLUTION);
  assert.equal(canvasSeedFor({ width: 512, height: 768 }, seeded, 1024), null);
});
