// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import ts from "typescript";
import {
  DEFAULT_RESOLUTION,
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

// The page's own size setters, seed, and load poll, run for real: the poll's ready branch is where
// a pick's canvas has to land, since the resident seed effect returns early for a load this page
// started.
function loadCanvasPage(opts: { pick: boolean; superseded?: boolean }) {
  const source = readSrc("features/images/images-page.tsx");
  const tree = ts.createSourceFile(
    "images-page.tsx",
    source,
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TSX,
  );
  const names = new Set([
    "canvasWrites",
    "seededCanvasWrites",
    "setWidth",
    "setHeight",
    "seedCanvas",
    "pollLoadProgress",
  ]);
  const declarations: string[] = [];
  function visit(node: ts.Node) {
    if (ts.isVariableDeclaration(node) && names.has(node.name.getText(tree))) {
      declarations.push(`const ${node.getText(tree)};`);
    }
    ts.forEachChild(node, visit);
  }
  visit(tree);
  assert.equal(declarations.length, names.size);
  const { outputText } = ts.transpileModule(
    `${declarations.join("\n")}\nreturn { setWidth, setHeight, seedCanvas, pollLoadProgress };`,
    { compilerOptions: { target: ts.ScriptTarget.ES2022 } },
  );
  const size = { width: 1024, height: 1024 };
  let recommended: number | null = null;
  const scope = {
    useCallback: (fn: unknown) => fn,
    useRef: (current: unknown) => ({ current }),
    setWidthState: (v: number) => {
      size.width = v;
    },
    setHeightState: (v: number) => {
      size.height = v;
    },
    setAspect: () => {},
    setPortrait: () => {},
    matchAspect: () => ({ key: "1:1", portrait: false }),
    resolutionFor,
    cancelSeq: { current: 0 },
    statusTicket: { current: 0 },
    getDiffusionLoadProgress: async () => ({ phase: "ready" }),
    getDiffusionStatus: async () => ({
      loaded: true,
      repo_id: "Qwen/Qwen-Image-2.1",
      recommended_canvas: recommended,
    }),
    dismissLoadToast: () => {},
    refreshStatus: async () => {},
    cancelLoadFromToast: () => {},
    setStatusIfNewest: () => {},
    toast: { success: () => {} },
    lastLoad: { current: { repoId: "Qwen/Qwen-Image-2.1", kind: "pipeline" } },
    matchesRememberedModel: () => false,
    rememberImageModel: () => {},
    setRememberedModel: () => {},
    quantRevert: {
      current: opts.pick ? { prev: null, steps: 20, guidance: 4 } : null,
    },
    pickRecipeSuperseded: { current: () => opts.superseded === true },
    setBusy: () => {},
    setPendingModelDefaults: () => {},
    lastLoadRevert: { current: null },
  };
  const page = new Function(...Object.keys(scope), outputText)(
    ...Object.values(scope),
  );
  return {
    size,
    page,
    ready: async (canvas: number | null) => {
      recommended = canvas;
      scope.quantRevert.current = opts.pick
        ? { prev: null, steps: 20, guidance: 4 }
        : null;
      await page.pollLoadProgress();
    },
  };
}

test("a model picked on this page gets the backend's canvas when its load lands", async () => {
  const { size, ready } = loadCanvasPage({ pick: true });
  await ready(512);
  assert.deepEqual(size, { width: 512, height: 512 });
  // The 512 was the page's own seed, so a roomier model picked next restores 1024.
  await ready(1024);
  assert.deepEqual(size, { width: 1024, height: 1024 });
});

test("a size the user wrote keeps its canvas, even at the seeded dimensions", async () => {
  // A restore or preset that lands on exactly the seeded 1024x1024 is still the user's.
  const { size, page, ready } = loadCanvasPage({ pick: true });
  page.setWidth(1024);
  page.setHeight(1024);
  await ready(512);
  assert.deepEqual(size, { width: 1024, height: 1024 });
});

test("a load no pick started, or a pick the user superseded, keeps its canvas", async () => {
  // Reapply and a recalled generation reload what the user already set up.
  const reapply = loadCanvasPage({ pick: false });
  await reapply.ready(512);
  assert.deepEqual(reapply.size, { width: 1024, height: 1024 });
  // A preset taken after the pick owns the form.
  const superseded = loadCanvasPage({ pick: true, superseded: true });
  await superseded.ready(512);
  assert.deepEqual(superseded.size, { width: 1024, height: 1024 });
});
