// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import ts from "@typescript/typescript6";

// Lifted from chat-adapter.ts source; importing it would drag in the stores and toast layer.
const adapterPath = fileURLToPath(
  new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
);
const source = readFileSync(adapterPath, "utf8");
const start = source.indexOf("function normalizeTarget(");
assert.ok(start >= 0, "normalizeTarget is no longer defined in chat-adapter.ts");
const declaration = source.slice(start, source.indexOf("\n}", start) + 2);
const normalizeTarget = new Function(
  `${ts.transpileModule(declaration, {
    compilerOptions: { target: ts.ScriptTarget.ES2020 },
  }).outputText}; return normalizeTarget;`,
)() as (value: string) => string;

const sameKey = (a: string, b: string) => normalizeTarget(a) === normalizeTarget(b);

test("one Windows file spelled with either separator is one candidate", () => {
  assert.ok(sameKey("C:\\Users\\a\\models\\M.gguf", "C:/Users/a/models/M.gguf"));
});

test("Windows and UNC paths still fold case", () => {
  assert.ok(sameKey("C:\\Users\\a\\M.gguf", "c:\\users\\a\\m.gguf"));
  assert.ok(sameKey("\\\\srv\\share\\M.gguf", "\\\\SRV\\share\\m.gguf"));
});

test("WSL UNC paths keep their case, because they address ext4", () => {
  assert.ok(
    !sameKey("\\\\wsl$\\Ubuntu\\home\\a\\M.gguf", "\\\\wsl$\\Ubuntu\\home\\a\\m.gguf"),
  );
  assert.ok(sameKey("\\\\wsl$\\Ubuntu\\home\\a\\M.gguf", "//wsl$/Ubuntu/home/a/M.gguf"));
});

test("POSIX paths keep their case", () => {
  assert.ok(!sameKey("/home/a/M.gguf", "/home/a/m.gguf"));
});

test("a decomposed filename is the same candidate as its composed form", () => {
  // macOS returns NFD filenames.
  assert.ok(sameKey("/home/a/caf\u00e9.gguf", "/home/a/cafe\u0301.gguf"));
});

test("repo ids still fold case", () => {
  assert.ok(sameKey("unsloth/Qwen3-0.6B-GGUF", "UNSLOTH/qwen3-0.6b-gguf"));
});


test("startup remembers an inactive quant through the logical chat inventory target", async () => {
  const declarations = ["buildAutoLoadSources", "resolveAutoLoadCandidate"].map((name) => {
    const begin = source.indexOf(`function ${name}(`);
    assert.ok(begin >= 0);
    const asyncPrefix = name === "resolveAutoLoadCandidate" ? "async " : "";
    return asyncPrefix + source.slice(begin, source.indexOf("\n}", begin) + 2);
  }).join("\n");
  const compile = new Function(
    "listGgufVariants", "isGgufLocalRow", "isAutoLoadableGgufVariant",
    ts.transpileModule(declarations, {
      compilerOptions: { target: ts.ScriptTarget.ES2020 },
    }).outputText + "; return { buildAutoLoadSources, resolveAutoLoadCandidate };",
  );
  const repoId = "Org/Model";
  const cachePath = "/cache/default/models--Org--Model";
  const scopes: (string | undefined)[] = [];
  const helpers = compile(
    async (_id: string, _token: undefined, options: { localPath?: string }) => {
      scopes.push(options.localPath);
      const quants = options.localPath === repoId ? ["Q6_K", "Q8_0"] : ["Q6_K"];
      return { variants: quants.map((quant, i) => ({
        quant, downloaded: true, size_bytes: 100 + i,
      })) };
    },
    () => true,
    () => true,
  );
  const rows = [{ repo_id: repoId, load_id: repoId, cache_path: cachePath, size_bytes: 300 }];
  const [logical] = helpers.buildAutoLoadSources(rows, [], [], 8192);
  const candidate = await helpers.resolveAutoLoadCandidate(logical, "Q8_0", () => false);
  assert.equal(candidate.ggufVariant, "Q8_0");
  assert.equal(candidate.loadId, repoId);
  assert.deepEqual(scopes, [repoId]);

  const snapshot = cachePath + "/snapshots/revision";
  const [explicit] = helpers.buildAutoLoadSources([{ ...rows[0], load_id: snapshot }], [], [], 8192);
  await explicit.listVariants();
  assert.equal(scopes.at(-1), snapshot);
  const [legacy] = helpers.buildAutoLoadSources([{ ...rows[0], load_id: undefined }], [], [], 8192);
  await legacy.listVariants();
  assert.equal(scopes.at(-1), cachePath);
  const localPath = "/models/local";
  const [local] = helpers.buildAutoLoadSources([], [], [{
    id: localPath, path: localPath, capabilities: { requires_variant: true },
  }], 8192);
  await local.listVariants();
  assert.equal(scopes.at(-1), localPath);
});

test("a remembered repo still ranks first when only its Hugging Face cache path is listed", () => {
  const declarations = [
    "normalizeTarget",
    "buildAutoLoadSources",
    "isRememberedSource",
    "orderAutoLoadSources",
  ].map((name) => {
    const begin = source.indexOf(`function ${name}(`);
    assert.ok(begin >= 0, `${name} is no longer defined in chat-adapter.ts`);
    return source.slice(begin, source.indexOf("\n}", begin) + 2);
  }).join("\n");
  const helpers = new Function(
    "listGgufVariants", "isGgufLocalRow",
    `${ts.transpileModule(declarations, {
      compilerOptions: { target: ts.ScriptTarget.ES2020 },
    }).outputText}; return { buildAutoLoadSources, orderAutoLoadSources };`,
  )(async () => ({ variants: [] }), () => true);
  const cacheRow = (repo: string, size: number) => {
    const path = `/home/a/.cache/huggingface/hub/models--${repo.replace("/", "--")}`;
    return {
      id: path, path, load_id: path, source: "hf_cache", model_id: repo,
      size_bytes: size, capabilities: { requires_variant: true },
    };
  };
  // The managed cache listing failed, so only /api/models/local rows name the cached repos.
  const sources = helpers.buildAutoLoadSources([], [], [
    cacheRow("unsloth/gemma-4-E2B-it-GGUF", 3_000_000_000),
    cacheRow("unsloth/gemma-4-26B-A4B-it-GGUF", 17_000_000_000),
  ], 8192);
  const remembered = { id: "unsloth/gemma-4-26B-A4B-it-GGUF", kind: "gguf" };
  const [first] = helpers.orderAutoLoadSources(sources, remembered);
  assert.equal(first.repoId, "unsloth/gemma-4-26B-A4B-it-GGUF");

  // Other local folders keep matching on their own target only.
  const studioRow = { ...cacheRow("unsloth/gemma-4-26B-A4B-it-GGUF", 17_000_000_000), source: "models_dir" };
  const [studioFirst] = helpers.orderAutoLoadSources(
    helpers.buildAutoLoadSources([], [], [
      cacheRow("unsloth/gemma-4-E2B-it-GGUF", 3_000_000_000),
      studioRow,
    ], 8192),
    remembered,
  );
  assert.equal(studioFirst.repoId, "unsloth/gemma-4-E2B-it-GGUF");
});
