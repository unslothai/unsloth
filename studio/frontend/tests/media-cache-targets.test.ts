// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import ts from "typescript";
import {
  createPickGuard,
  runGgufRepoPick,
} from "../src/lib/diffusion-gguf-pick.ts";
import { diffusionRoutePick } from "../src/lib/diffusion-route-pick.ts";
import { diffusionRouteSearch } from "../src/lib/diffusion-route-search.ts";

const repoId = "unsloth/Z-Image-Turbo-GGUF";
const localPath =
  "/custom/models--unsloth--Z-Image-Turbo-GGUF/snapshots/revision";
const filename = "model-Q8_0.gguf";

function parse(path: string) {
  return ts.createSourceFile(
    path,
    readFileSync(new URL(`../src/${path}`, import.meta.url), "utf8"),
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TSX,
  );
}
function find(
  file: ts.SourceFile,
  predicate: (node: ts.Node) => boolean,
): ts.Node[] {
  const found: ts.Node[] = [];
  function visit(node: ts.Node) {
    if (predicate(node)) found.push(node);
    ts.forEachChild(node, visit);
  }
  visit(file);
  return found;
}
function evaluate(
  node: ts.Node,
  file: ts.SourceFile,
  env: Record<string, unknown>,
) {
  const js = ts.transpileModule(`const result = (${node.getText(file)});`, {
    compilerOptions: { target: ts.ScriptTarget.ES2020 },
  }).outputText;
  return new Function(...Object.keys(env), `${js}; return result;`)(
    ...Object.values(env),
  );
}
function callback(
  file: ts.SourceFile,
  name: string,
  env: Record<string, unknown>,
) {
  const [decl] = find(
    file,
    (node) =>
      ts.isVariableDeclaration(node) && node.name.getText(file) === name,
  ) as ts.VariableDeclaration[];
  assert.ok(decl?.initializer && ts.isCallExpression(decl.initializer), name);
  return evaluate(decl.initializer.arguments[0], file, env);
}

for (const page of ["images", "video"]) {
  test(`${page}: route validation and exact filename keep the physical target`, () => {
    const file = parse(`app/routes/${page}.tsx`);
    const [property] = find(
      file,
      (node) =>
        ts.isPropertyAssignment(node) &&
        node.name.getText(file) === "validateSearch",
    ) as ts.PropertyAssignment[];
    const validate = evaluate(property.initializer, file, {});
    const search = validate(
      diffusionRouteSearch(repoId, {
        loadId: localPath,
        ggufFilename: filename,
      }),
    );
    const pick = diffusionRoutePick(
      search.model,
      search.quant,
      undefined,
      search.loadId,
    );
    assert.equal(pick.repoId, repoId);
    assert.equal(pick.opts.localPath, localPath);
    assert.equal(pick.opts.filename, filename);
  });

  test(`${page}: quant resolution, planning, loading, and Reapply keep the cache target`, async () => {
    const file = parse(`features/${page}/${page}-page.tsx`);
    const calls: unknown[][] = [];
    const pick = callback(file, "loadGgufRepoPick", {
      pickGuard: createPickGuard(),
      isMounted: { current: true },
      quantRevert: { current: null },
      quant: null,
      steps: 1,
      guidance: 1,
      runGgufRepoPick,
      hfApiToken: () => undefined,
      getHfToken: () => undefined,
      resolveDiffusionGgufFilename: async (
        _id: string,
        opts: { localPath?: string },
      ) => {
        assert.equal(opts.localPath, localPath);
        return filename;
      },
      toast: { error: assert.fail },
      setQuant: () => {},
      revertPick: () => {},
      applyImageModelDefaults: () => {},
      applyVideoModelDefaults: () => {},
      loadOrStage: async (...args: unknown[]) => {
        calls.push(args);
        return true;
      },
    });
    assert.equal(await pick(repoId, "Q8_0", "hub", localPath), true);
    assert.deepEqual(calls[0].slice(0, 3), [
      repoId,
      { kind: "gguf", filename, localPath },
      "hub",
    ]);
    const opts = calls[0][1];
    const paths = find(
      file,
      (node) =>
        ts.isPropertyAssignment(node) &&
        node.name.getText(file) === "model_path",
    ) as ts.PropertyAssignment[];
    assert.ok(paths.length >= 2);
    for (const property of paths) {
      assert.equal(
        evaluate(property.initializer, file, {
          repoId,
          opts,
          meta: { loadId: localPath },
        }),
        localPath,
      );
    }
    const reapply = callback(file, "handleReapply", {
      lastLoad: { current: { repoId, ...(opts as object) } },
      handleLoad: (...args: unknown[]) => calls.push(args),
    });
    reapply();
    assert.equal(
      (calls.at(-1)?.[1] as { localPath: string }).localPath,
      localPath,
    );
  });
}
