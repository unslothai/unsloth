// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import ts from "typescript";
import { settingsGgufVariantForRow } from "../src/features/hub/inventory/settings-identity.ts";
import { ggufVariantsMatch, modelIdsMatch } from "../src/features/hub/lib/model-identity.ts";

const text = readFileSync(new URL("../src/features/hub/hub-page.tsx", import.meta.url), "utf8");
const source = ts.createSourceFile("hub-page.tsx", text, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
let expression: string | undefined;
function visit(node: ts.Node) {
  if (ts.isVariableDeclaration(node) && node.name.getText(source) === "openModelSettings") {
    assert.ok(node.initializer && ts.isCallExpression(node.initializer));
    expression = node.initializer.arguments[0].getText(source);
  }
  ts.forEachChild(node, visit);
}
visit(source);
assert.ok(expression);
const js = ts.transpileModule(`const callback = ${expression};`, {
  compilerOptions: { target: ts.ScriptTarget.ES2020 },
}).outputText;

for (const resident of ["Q6_K", "Q8_0"]) {
  for (const [kind, pipelineTag] of [
    ["cache", null], ["cache", "text-generation"], ["cache", "image-text-to-text"],
    ["cache", "text-to-image"], ["local", "text-generation"],
  ] as const) {
    test(`${kind} ${pipelineTag ?? "unknown chat"} settings retain ${resident} after a folder switch`, async () => {
      const repoId = "Org/Model-GGUF";
      const physicalPath = "/cache/preferred/models--Org--Model-GGUF";
      const chat = kind === "cache" && pipelineTag !== "text-to-image";
      const localQuant = resident === "Q6_K" ? "Q8_0" : "Q6_K";
      let target: { ggufVariant: string; id: string } | undefined;
      let request: { localPath?: string | null; includeCacheLocations?: boolean } | undefined;
      const row = {
        kind, pipelineTag, loadId: chat ? repoId : physicalPath,
        repoId, cachePath: physicalPath, path: physicalPath, isGguf: true,
        capabilities: { requiresVariant: true }, partial: false,
      };
      const context = {
        settingsOpenSeq: { current: 0 }, refreshResidentModelStatus: async () => {},
        settingsGgufVariantForRow, hfApiToken: (value: string) => value,
        hfToken: "hf_fixture_token", ggufVariantsMatch, modelIdsMatch,
        isChatGgufTask: (task: string | null) => !task || ["text-generation", "image-text-to-text"].includes(task),
        isExternalModelId: () => false,
        useChatRuntimeStore: { getState: () => ({ params: { checkpoint: row.loadId }, activeGgufVariant: resident }) },
        listGgufVariants: async (_repo: string, token: string, options: typeof request) => {
          assert.equal(token, "hf_fixture_token");
          request = options;
          const logical = options?.localPath === repoId && options?.includeCacheLocations !== false;
          return {
            variants: (logical ? [localQuant, resident] : [localQuant]).map((quant) => ({ quant, downloaded: true })),
            default_variant: localQuant,
          };
        },
        setSettingsTarget: (value: typeof target) => { target = value; },
        LOCAL_MODEL_SOURCE: { OLLAMA: "ollama" },
        toast: { error: () => assert.fail("settings should resolve a downloaded quant") },
      };
      const callback = new Function(...Object.keys(context), js + "; return callback;")(...Object.values(context));
      await callback(row);
      assert.equal(target?.ggufVariant, chat ? resident : localQuant);
      assert.equal(target?.id, row.loadId);
      assert.equal(request?.localPath, chat ? repoId : physicalPath);
    });
  }
}
