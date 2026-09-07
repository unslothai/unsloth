// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import ts from "typescript";
import {
  isHfCacheSnapshotPath,
  publicModelId,
} from "../src/features/hub/lib/model-identity.ts";
import { modelIdsMatchForPicker } from "../src/features/model-picker/components/model-selector/row-identity.ts";
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
const hfCacheRepoId = (path: string) =>
  isHfCacheSnapshotPath(path) ? publicModelId(path) : null;

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
  const js = ts.transpileModule(
    `const result = (${node.getText(file).replace(/^export\s+/, "")});`,
    {
      compilerOptions: { target: ts.ScriptTarget.ES2020 },
    },
  ).outputText;
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
  test(`${page}: hydrated status marks the logical repository and exact resident copy`, () => {
    const file = parse(`features/${page}/${page}-page.tsx`);
    const status = { loaded: true, repo_id: localPath, gguf_variant: "Q8_0" };
    const env: Record<string, unknown> = {
      status,
      quant: "Q6_K",
      hfCacheRepoId,
    };
    for (const name of ["residentLoadId", "residentModelId"]) {
      const [decl] = find(
        file,
        (n) => ts.isVariableDeclaration(n) && n.name.getText(file) === name,
      ) as ts.VariableDeclaration[];
      if (decl?.initializer) env[name] = evaluate(decl.initializer, file, env);
    }
    const [selector] = find(
      file,
      (n) =>
        ts.isJsxSelfClosingElement(n) &&
        n.tagName.getText(file) === "ModelSelector",
    ) as ts.JsxSelfClosingElement[];
    const props = Object.fromEntries(
      selector.attributes.properties.flatMap((p) => {
        if (
          !ts.isJsxAttribute(p) ||
          !p.initializer ||
          !ts.isJsxExpression(p.initializer) ||
          !p.initializer.expression
        )
          return [];
        const name = p.name.getText(file);
        if (
          ![
            "value",
            "selectedLoadId",
            "selectedGgufVariant",
            "loadedModelIdOverride",
            "loadedLoadIdOverride",
            "loadedGgufVariantOverride",
          ].includes(name)
        )
          return [];
        return [[name, evaluate(p.initializer.expression, file, env)]];
      }),
    );
    assert.equal(props.value, repoId);
    assert.equal(props.selectedLoadId, localPath);
    assert.equal(props.selectedGgufVariant, "Q8_0");
    assert.equal(props.loadedModelIdOverride, repoId);
    assert.equal(props.loadedLoadIdOverride, localPath);
    assert.equal(props.loadedGgufVariantOverride, "Q8_0");
  });
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

test("chat status keeps its logical checkpoint and physical load identity", () => {
  const file = parse("features/chat/lib/apply-inference-status-to-store.ts");
  const [fn] = find(
    file,
    (n) =>
      ts.isFunctionDeclaration(n) &&
      n.name?.text === "resolveInferenceCheckpointId",
  ) as ts.FunctionDeclaration[];
  const resolve = evaluate(fn, file, { hfCacheRepoId });
  const status = {
    active_model: repoId,
    model_identifier: localPath,
    is_gguf: true,
  };
  assert.equal(resolve(status), repoId);
  const [state] = find(
    file,
    (n) =>
      ts.isObjectLiteralExpression(n) &&
      n.properties.some(
        (p) =>
          ts.isPropertyAssignment(p) &&
          p.name.getText(file) === "residentCheckpoint",
      ),
  ) as ts.ObjectLiteralExpression[];
  assert.deepEqual(evaluate(state, file, { status, checkpointId: repoId }), {
    residentCheckpoint: repoId,
    activeLoadId: localPath,
  });
  assert.deepEqual(evaluate(state, file, {
    status: { ...status, model_identifier: repoId, cache_load_id: localPath }, checkpointId: repoId,
  }), { residentCheckpoint: repoId, activeLoadId: localPath });
});

test("the transcription run picker excludes unsupported inactive snapshots", () => {
  const file = parse(
    "features/model-picker/components/model-selector/pickers.tsx",
  );
  const [decl] = find(
    file,
    (n) =>
      ts.isVariableDeclaration(n) &&
      n.name.getText(file) === "sortedCachedGguf",
  ) as ts.VariableDeclaration[];
  assert.ok(decl.initializer && ts.isCallExpression(decl.initializer));
  for (const task of ["automatic-speech-recognition", undefined]) {
    const rows: unknown[] = evaluate(decl.initializer.arguments[0], file, {
      cachedGguf: [
        { repo_id: "Org/STT", active_cache: true },
        { repo_id: "Org/STT", active_cache: false },
      ],
      task,
      catalog: [],
      activeCatalogArtifactIds: [],
      downloadedSort: "name",
      loadTimes: {},
      sortCachedRepos: (candidates: unknown[]) => candidates,
      passesTaskGate: () => true,
      audioPickIsRoutable: () => true,
      artifactForRepoId: () => ({}),
      AUDIO_CATALOG: [],
    })();
    assert.equal(rows.length, task ? 1 : 2);
  }
});

test("media residency overrides mark only the loaded cache copy", () => {
  const file = parse(
    "features/model-picker/components/model-selector/pickers.tsx",
  );
  const [decl] = find(
    file,
    (n) =>
      ts.isVariableDeclaration(n) &&
      n.name.getText(file) === "matchesLoadedCacheCopy",
  ) as ts.VariableDeclaration[];
  assert.ok(decl.initializer);
  const matches = evaluate(decl.initializer, file, {
    loadedModelIdOverride: repoId,
    loadedLoadIdOverride: localPath,
    loadedModelId: repoId,
    activeLoadId: localPath,
    modelIdsMatchForPicker,
  });
  assert.equal(matches(repoId, localPath), true);
  assert.equal(
    matches(
      repoId,
      "/default/models--unsloth--Z-Image-Turbo-GGUF/snapshots/other",
    ),
    false,
  );
});
